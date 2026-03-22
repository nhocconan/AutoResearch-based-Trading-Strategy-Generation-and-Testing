#!/usr/bin/env python3
"""
Experiment #402: 1d Daily Regime-Adaptive with Weekly Trend Bias

Hypothesis: Daily timeframe captures major moves while avoiding intraday noise.
After 401 failed experiments, the key insight is that 1d strategies need:
1. WEEKLY trend bias (via mtf_data) to avoid trading against major trend
2. REGIME detection (Choppiness or BB Width) to switch between trend/mean-revert
3. LOOSE entry conditions to generate sufficient trades on daily data
4. ATR stoploss (2.5x) to survive 2022-style crashes

STRATEGY COMPONENTS:
1. 1w HMA(21) via mtf_data: Major trend bias (long only when price > 1w HMA)
2. Choppiness Index(14): Regime detection (>61.8 = range, <38.2 = trend)
3. RSI(14) + Z-score(20): Entry signals for both regimes
4. Donchian(20) breakout: Trend continuation entries
5. ATR(14) trailing stop: 2.5x for risk management

Why this should work on 1d:
- Weekly trend filter prevents counter-trend trades in major moves
- Regime detection adapts to market conditions (trend vs range)
- Multiple entry types ensure sufficient trade frequency
- Daily bars = less noise, cleaner signals
- Position sizing 0.30 discrete limits drawdown

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_weekly_hma_rsi_zscore_donchian_atr_v1"
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
    return rsi

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean-reversion signals."""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean().values
    rolling_std = close_s.rolling(window=period, min_periods=period).std().values
    zscore = (close - rolling_mean) / (rolling_std + 1e-10)
    return zscore

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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bounds."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = high[i-period+1:i+1].max()
        lower[i] = low[i-period+1:i+1].min()
    
    return upper, lower

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
    zscore = calculate_zscore(close, 20)
    chop = calculate_choppiness_index(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
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
        
        if np.isnan(rsi[i]) or np.isnan(zscore[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY TREND BIAS ===
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION ===
        ranging_market = chop[i] > 61.8
        trending_market = chop[i] < 38.2
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        # TRENDING REGIME: Donchian breakout + weekly trend confirmation
        if trending_market:
            # Long breakout above Donchian upper + weekly bullish
            if close[i] > donchian_upper[i - 1] and bull_trend_1w:
                new_signal = SIZE
            # Short breakout below Donchian lower + weekly bearish
            elif close[i] < donchian_lower[i - 1] and bear_trend_1w:
                new_signal = -SIZE
            # RSI pullback entries in trend direction
            elif bull_trend_1w and rsi[i] < 45 and rsi[i] > 30:
                new_signal = SIZE
            elif bear_trend_1w and rsi[i] > 55 and rsi[i] < 70:
                new_signal = -SIZE
        
        # RANGING REGIME: Z-score mean-reversion
        elif ranging_market:
            # Long when Z-score < -1.5 (oversold)
            if zscore[i] < -1.5:
                new_signal = SIZE
            # Short when Z-score > +1.5 (overbought)
            elif zscore[i] > 1.5:
                new_signal = -SIZE
            # RSI extremes for additional entries
            elif rsi[i] < 35:
                new_signal = SIZE
            elif rsi[i] > 65:
                new_signal = -SIZE
        
        # NEUTRAL REGIME (38.2 <= CHOP <= 61.8): Only Z-score extremes
        else:
            if zscore[i] < -2.0:
                new_signal = SIZE
            elif zscore[i] > 2.0:
                new_signal = -SIZE
        
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
            # Long position should exit if weekly trend turns bearish
            if position_side > 0 and bear_trend_1w:
                new_signal = 0.0
            # Short position should exit if weekly trend turns bullish
            if position_side < 0 and bull_trend_1w:
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