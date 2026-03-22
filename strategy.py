#!/usr/bin/env python3
"""
Experiment #424: 4h Choppiness Regime + 1d HMA Trend + Adaptive Entry

Hypothesis: After analyzing 423 failed experiments, the key insight is that ADX
is inferior to Choppiness Index (CHOP) for regime detection on 4h timeframe.
CHOP specifically measures market choppiness vs trending, while ADX only measures
trend strength. This strategy uses:

1. CHOPPINESS INDEX (14) REGIME DETECTION on 4h:
   - CHOP > 61.8 = ranging market (mean reversion entries)
   - CHOP < 38.2 = trending market (breakout entries)
   - 38.2-61.8 = neutral (stay flat or reduce position)
   - This is MORE accurate than ADX for crypto markets

2. 1d HMA(21) TREND BIAS (via mtf_data helper):
   - Long bias when price > 1d HMA
   - Short bias when price < 1d HMA
   - HMA reduces lag vs EMA, critical for MTF alignment

3. ADAPTIVE ENTRY LOGIC:
   - Trending: Donchian(20) breakout with trend confirmation
   - Ranging: RSI(14) extremes (25/75) with mean reversion
   - Asymmetric thresholds: easier to enter WITH trend, harder AGAINST

4. 1w HMA(21) META-FILTER:
   - Only take longs when price > 1w HMA (bullish meta-trend)
   - Only take shorts when price < 1w HMA (bearish meta-trend)
   - Reduces counter-trend trades that get destroyed in 2022

5. ATR(14) TRAILING STOP at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Protects from crashes while allowing normal volatility

6. POSITION SIZING: 0.25 discrete (conservative for 4h)
   - Max 25% capital per position
   - Discrete levels: 0.0, ±0.25 only (minimize fee churn)

Why 4h should work:
- Enough trades (~50-100/year) for statistical significance
- Less noise than 1h/15m, more signals than 12h/1d
- Good balance for regime detection
- Should work on BTC/ETH/SOL individually (not SOL-biased)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_1d_1w_hma_adaptive_atr_v2"
timeframe = "4h"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_series = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    
    for i in range(period, n):
        sum_atr = atr_series[i-period+1:i+1].sum()
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 0:
            chop[i] = 100 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
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
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        trending_market = chop[i] < 38.2
        ranging_market = chop[i] > 61.8
        # neutral_market = 38.2 <= CHOP <= 61.8
        
        # === 1d HMA TREND BIAS ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === 1w HMA META-FILTER ===
        bull_meta_1w = close[i] > hma_1w_aligned[i]
        bear_meta_1w = close[i] < hma_1w_aligned[i]
        
        # === DONCHIAN BREAKOUT SIGNALS (for trending regime) ===
        donchian_long = close[i] > donchian_upper[i-1]  # Break above previous high
        donchian_short = close[i] < donchian_lower[i-1]  # Break below previous low
        
        # === RSI MEAN REVERSION SIGNALS (for ranging regime) ===
        rsi_long = rsi[i] < 25  # Deeply oversold
        rsi_short = rsi[i] > 75  # Deeply overbought
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # TRENDING REGIME: Donchian breakout with 1d/1w HMA filter
        if trending_market:
            # Long: bullish 1d + bullish 1w + breakout
            if bull_trend_1d and bull_meta_1w and donchian_long:
                new_signal = SIZE
            # Short: bearish 1d + bearish 1w + breakout
            elif bear_trend_1d and bear_meta_1w and donchian_short:
                new_signal = -SIZE
            # Weaker signals: only 1d confirmation (reduce size)
            elif bull_trend_1d and donchian_long:
                new_signal = SIZE / 2  # 0.125
            elif bear_trend_1d and donchian_short:
                new_signal = -SIZE / 2  # -0.125
        
        # RANGING REGIME: RSI mean-reversion with 1d HMA filter
        elif ranging_market:
            # Long: oversold + 1d bullish bias (easier with trend)
            if rsi_long and bull_trend_1d:
                new_signal = SIZE
            # Long: deeply oversold even against 1d trend (stronger signal needed)
            elif rsi[i] < 20 and bear_trend_1d:
                new_signal = SIZE / 2  # Counter-trend needs stronger RSI
            # Short: overbought + 1d bearish bias
            elif rsi_short and bear_trend_1d:
                new_signal = -SIZE
            # Short: deeply overbought even against 1d trend
            elif rsi[i] > 80 and bull_trend_1d:
                new_signal = -SIZE / 2  # Counter-trend needs stronger RSI
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if high[i] > highest_price:
                    highest_price = high[i]
                stoploss_price = highest_price - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or low[i] < lowest_price:
                    lowest_price = low[i]
                stoploss_price = lowest_price + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME FLIP EXIT ===
        # Exit if regime changes and position is counter to new regime logic
        if in_position and new_signal != 0.0:
            # Long in trending should exit if market becomes ranging without RSI support
            if position_side > 0 and ranging_market and not rsi_long and rsi[i] > 40:
                new_signal = 0.0
            # Short in trending should exit if market becomes ranging without RSI support
            if position_side < 0 and ranging_market and not rsi_short and rsi[i] < 60:
                new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            # Long position exits if 1d trend flips bearish
            if position_side > 0 and bear_trend_1d and not ranging_market:
                new_signal = 0.0
            # Short position exits if 1d trend flips bullish
            if position_side < 0 and bull_trend_1d and not ranging_market:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = high[i] if position_side > 0 else 0.0
                lowest_price = low[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = high[i] if position_side > 0 else 0.0
                lowest_price = low[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals