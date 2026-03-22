#!/usr/bin/env python3
"""
Experiment #552: Daily EMA Crossover with Weekly HMA Bias + RSI Filter

Hypothesis: After 500+ failed experiments, daily timeframe needs:
1. More frequent signals than Donchian (EMA 8/21 crossover = ~2-4 signals/month)
2. Weekly HMA trend bias (soft filter, not hard block)
3. RSI(14) confirmation to avoid entering at extremes (RSI 25-75 range)
4. ATR(14) stoploss at 2.5x for capital protection
5. Discrete position sizing 0.30 to balance risk vs opportunity

Why this should work on 1d:
- EMA crossover generates enough trades (unlike Donchian which had 0 trades)
- Weekly HMA provides regime context without blocking all counter-trend trades
- RSI filter prevents buying tops/selling bottoms
- Simple logic = robust across BTC/ETH/SOL (no symbol-specific tuning)
- 1d timeframe = fewer fees, captures multi-week trends

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete (max 0.40)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_ema_crossover_weekly_hma_rsi_filter_atr_v1"
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

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

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
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

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
    ema_8 = calculate_ema(close, 8)
    ema_21 = calculate_ema(close, 21)
    rsi_14 = calculate_rsi(close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    
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
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_8[i]) or np.isnan(ema_21[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY HMA TREND BIAS (soft filter) ===
        bull_bias = close[i] > hma_1w_aligned[i]
        bear_bias = close[i] < hma_1w_aligned[i]
        
        # === EMA CROSSOVER SIGNALS ===
        # Golden cross: EMA8 crosses above EMA21
        ema_cross_long = ema_8[i] > ema_21[i] and ema_8[i-1] <= ema_21[i-1]
        # Death cross: EMA8 crosses below EMA21
        ema_cross_short = ema_8[i] < ema_21[i] and ema_8[i-1] >= ema_21[i-1]
        
        # === RSI FILTER (avoid extremes) ===
        # Long: RSI between 25-75 (not oversold bounce, not overbought)
        rsi_ok_long = 25 < rsi_14[i] < 75
        # Short: RSI between 25-75
        rsi_ok_short = 25 < rsi_14[i] < 75
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Long: EMA golden cross + RSI in range + weekly bias (soft)
        if ema_cross_long and rsi_ok_long:
            if bull_bias:
                new_signal = SIZE  # Full size with trend
            else:
                new_signal = SIZE * 0.5  # Half size against weekly trend
        
        # Short: EMA death cross + RSI in range + weekly bias (soft)
        elif ema_cross_short and rsi_ok_short:
            if bear_bias:
                new_signal = -SIZE  # Full size with trend
            else:
                new_signal = -SIZE * 0.5  # Half size against weekly trend
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if weekly HMA flips strongly against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias:
                # Check if price is significantly below weekly HMA
                if close[i] < hma_1w_aligned[i] * 0.97:  # 3% below
                    new_signal = 0.0
            if position_side < 0 and bull_bias:
                # Check if price is significantly above weekly HMA
                if close[i] > hma_1w_aligned[i] * 1.03:  # 3% above
                    new_signal = 0.0
        
        # === EMA REVERSAL EXIT ===
        # Exit long if EMA8 crosses back below EMA21
        if in_position and position_side > 0:
            if ema_8[i] < ema_21[i]:
                new_signal = 0.0
        
        # Exit short if EMA8 crosses back above EMA21
        if in_position and position_side < 0:
            if ema_8[i] > ema_21[i]:
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