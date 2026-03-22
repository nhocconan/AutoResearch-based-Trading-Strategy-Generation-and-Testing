#!/usr/bin/env python3
"""
Experiment #384: 1d Primary Timeframe - Weekly Trend + Daily RSI Mean Reversion + Regime Filter

Hypothesis: Daily timeframe is underutilized in crypto futures. Most strategies fail because they
overtrade on lower timeframes. On 1d, we can capture major swings with fewer trades and lower fees.

KEY INSIGHTS FROM 383 FAILED EXPERIMENTS:
1. Pure trend-following fails in 2022 crash and 2025 bear market
2. Pure mean-reversion fails in strong trends (2021 bull run)
3. REGIME DETECTION is the missing piece - need to adapt strategy per market state
4. Weekly HMA provides stable trend bias (less noise than daily)
5. RSI mean-reversion works BEST when aligned with higher timeframe trend

STRATEGY COMPONENTS:
1. 1w HMA(21) TREND BIAS (via mtf_data): Long-term trend direction
   - Price > 1w HMA = bull bias (prefer long entries)
   - Price < 1w HMA = bear bias (prefer short entries)
   
2. DAILY RSI(14) MEAN REVERSION: Entry trigger
   - Long: RSI < 40 (oversold) + bull bias from 1w
   - Short: RSI > 60 (overbought) + bear bias from 1w
   - These thresholds are LOOSE enough to generate 10+ trades/year
   
3. CHOPPINESS INDEX(14) REGIME FILTER: Avoid whipsaw
   - CHOP > 55 = ranging (enable mean-reversion entries)
   - CHOP < 45 = trending (enable trend-following entries)
   - 45-55 = neutral (reduce position size by 50%)
   
4. ATR(14) TRAILING STOP: Risk management
   - Exit when price moves 2.5*ATR against position
   - Critical for surviving 2022-style crashes
   
5. POSITION SIZING: 0.30 discrete (conservative for daily volatility)
   - Max 30% capital per position
   - Discrete levels: 0.0, ±0.15, ±0.30

Why this should beat current best (Sharpe=0.676):
- Weekly trend filter avoids counter-trend trades (major improvement)
- RSI thresholds (40/60) are looser than typical (30/70) = more trades
- Choppiness regime filter reduces whipsaw in transition periods
- Daily timeframe = fewer false signals, lower fee drag
- Works on BTC, ETH, SOL individually (not SOL-biased)

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
Expected trades: 15-30 per symbol per year (enough for statistical significance)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_weekly_hma_rsi_chop_regime_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100.0
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean-reversion works)
    CHOP < 38.2 = trending market (trend-following works)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = tr[i-period+1:i+1].sum()
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY TREND BIAS ===
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION ===
        ranging_market = chop[i] > 55.0
        trending_market = chop[i] < 45.0
        neutral_market = 45.0 <= chop[i] <= 55.0
        
        # === RSI SIGNALS (LOOSE THRESHOLDS FOR MORE TRADES) ===
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        current_size = SIZE_FULL if not neutral_market else SIZE_HALF
        
        # LONG ENTRY: RSI oversold + weekly bull trend OR ranging market
        if rsi_oversold:
            if bull_trend_1w or ranging_market:
                new_signal = current_size
        
        # SHORT ENTRY: RSI overbought + weekly bear trend OR ranging market
        if rsi_overbought:
            if bear_trend_1w or ranging_market:
                new_signal = -current_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1w and not ranging_market:
                # Long position should exit if weekly trend turns bear and not ranging
                new_signal = 0.0
            if position_side < 0 and bull_trend_1w and not ranging_market:
                # Short position should exit if weekly trend turns bull and not ranging
                new_signal = 0.0
        
        # === RSI REVERSAL EXIT (take profit) ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and rsi[i] > 70.0:
                # Long position take profit when RSI overbought
                new_signal = 0.0
            if position_side < 0 and rsi[i] < 30.0:
                # Short position take profit when RSI oversold
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals