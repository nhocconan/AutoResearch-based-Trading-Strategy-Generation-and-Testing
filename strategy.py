#!/usr/bin/env python3
"""
Experiment #017: 1d HMA Trend + 1W Bias + Volatility Regime + RSI Pullback

Hypothesis: After 16 failed experiments, the clear pattern is:
1. Complex multi-filter strategies generate 0 trades (too many conflicting conditions)
2. Lower timeframes suffer from noise and fee drag
3. Simple trend + pullback works better than complex regime switching
4. 1d timeframe should have optimal trade frequency (20-50/year)

This strategy SIMPLIFIES entry conditions to ensure trades generate:

1. 1W HMA trend bias: Only long if price > 1w_HMA, only short if price < 1w_HMA
   (Ultra-stable HTF filter - call get_htf_data ONCE before loop)

2. 1d HMA crossover: Fast HMA(10) vs Slow HMA(30) for trend direction
   Simpler than Donchian, less lag than EMA

3. RSI pullback entry: RSI(14) between 35-55 for long, 45-65 for short
   LOOSENED from extreme values to ensure trades generate

4. Volatility regime: ATR(14)/ATR(50) ratio for position sizing
   High vol = smaller size, low vol = normal size

5. ATR trailing stop: 2.5 * ATR(14) to protect from crashes

Key difference from failed strategies:
- FEWER filters (removed Choppiness, CRSI, Funding - all caused 0 trades)
- LOOSER RSI thresholds (35-55 instead of <15 or >85)
- No complex regime switching (just HTF bias + trend + pullback)
- Target 30-50 trades/year on 1d

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_trend_1w_bias_rsi_pullback_vol_atr_v1"
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
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

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
    hma_fast = calculate_hma(close, 10)
    hma_slow = calculate_hma(close, 30)
    rsi_14 = calculate_rsi(close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    atr_50 = calculate_atr(high, low, close, 50)
    
    # Volatility ratio for position sizing
    vol_ratio = atr_14 / np.where(atr_50 > 0, atr_50, atr_14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.30
    HIGH_VOL_SIZE = 0.20  # Reduce size in high volatility
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        # === 1W HMA TREND BIAS (Ultra-stable HTF filter) ===
        bull_bias = close[i] > hma_1w_aligned[i]
        bear_bias = close[i] < hma_1w_aligned[i]
        
        # === 1D HMA TREND ===
        hma_bullish = hma_fast[i] > hma_slow[i]
        hma_bearish = hma_fast[i] < hma_slow[i]
        
        # === RSI PULLBACK (LOOSENED thresholds to ensure trades) ===
        # Long: RSI between 35-55 (pullback in uptrend, not extreme)
        # Short: RSI between 45-65 (rally in downtrend, not extreme)
        rsi_long_pullback = 35 <= rsi_14[i] <= 55
        rsi_short_pullback = 45 <= rsi_14[i] <= 65
        
        # === VOLATILITY REGIME ===
        is_high_vol = vol_ratio[i] > 1.5 if not np.isnan(vol_ratio[i]) else False
        
        # === POSITION SIZING ===
        base_size = HIGH_VOL_SIZE if is_high_vol else BASE_SIZE
        
        # === ENTRY LOGIC (SIMPLIFIED - fewer filters) ===
        new_signal = 0.0
        
        # Long entry: 1W bullish + 1D HMA bullish + RSI pullback
        if bull_bias and hma_bullish and rsi_long_pullback:
            new_signal = base_size
        
        # Short entry: 1W bearish + 1D HMA bearish + RSI pullback
        elif bear_bias and hma_bearish and rsi_short_pullback:
            new_signal = -base_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_exit = False
        if in_position and position_side != 0:
            # Exit if 1D HMA crosses against position
            if position_side > 0 and hma_bearish:
                trend_exit = True
            if position_side < 0 and hma_bullish:
                trend_exit = True
        
        # Apply stoploss or trend exit
        if stoploss_triggered or trend_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals