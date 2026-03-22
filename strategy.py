#!/usr/bin/env python3
"""
Experiment #041: 12h Asymmetric Trend with 1d HMA Regime + RSI Pullback
Hypothesis: 12h timeframe captures intermediate trends while avoiding 1h/4h noise.
Key insight: Previous 12h strategies failed due to too many conflicting filters (0 trades) or symmetric longs/shorts (whipsaw in 2022).
This strategy uses ASYMMETRIC logic: aggressive longs when 1d HMA bullish, conservative shorts when 1d HMA bearish.
Entry: 12h EMA21>EMA50 crossover + RSI pullback (40-60 range) + price above 1d HMA for longs.
Exit: ATR trailing stop (2.5*ATR) + signal reversal.
Position sizing: 0.30 for longs, 0.20 for shorts (asymmetric risk).
Why this might work: Fewer filters = more trades. Asymmetric sizing controls drawdown in bear markets.
Timeframe: 12h (REQUIRED), HTF: 1d via mtf_data helper (call ONCE before loop).
Must generate 10+ trades on train, 3+ on test - entry conditions deliberately loosened.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_asym_trend_1d_hma_rsi_v2"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_keltner(high, low, close, ema_period=20, atr_period=10, mult=2.0):
    """Calculate Keltner Channels for volatility-based entries."""
    ema = calculate_ema(close, ema_period)
    atr = calculate_atr(high, low, close, atr_period)
    upper = ema + mult * atr
    lower = ema - mult * atr
    return upper, lower, ema

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    sma_50 = calculate_sma(close, 50)
    
    # Keltner channels for volatility breakouts
    keltner_upper, keltner_lower, keltner_mid = calculate_keltner(high, low, close, 20, 10, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing - asymmetric (Rule 4)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.20
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(ema_50[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF) - main regime filter
        bull_regime = close[i] > hma_1d_aligned[i]
        bear_regime = close[i] < hma_1d_aligned[i]
        
        # 12h trend confirmation - LOOSENED conditions
        bull_trend_12h = ema_21[i] > ema_50[i]
        bear_trend_12h = ema_21[i] < ema_50[i]
        
        # Long-term trend filter
        above_200 = not np.isnan(ema_200[i]) and close[i] > ema_200[i]
        below_200 = not np.isnan(ema_200[i]) and close[i] < ema_200[i]
        
        # RSI conditions - LOOSENED for more trades (key fix from failures)
        rsi_ok_long = 35 < rsi[i] < 65  # Wide range
        rsi_ok_short = 35 < rsi[i] < 65
        
        # Price momentum - simple higher low / lower high
        higher_low = False
        lower_high = False
        if i >= 5:
            higher_low = low[i] > low[i-5]
            lower_high = high[i] < high[i-5]
        
        # Keltner breakout
        keltner_breakout_long = close[i] > keltner_upper[i]
        keltner_breakout_short = close[i] < keltner_lower[i]
        
        # EMA crossover signal
        ema_cross_long = False
        ema_cross_short = False
        if i >= 1:
            ema_cross_long = ema_21[i] > ema_50[i] and ema_21[i-1] <= ema_50[i-1]
            ema_cross_short = ema_21[i] < ema_50[i] and ema_21[i-1] >= ema_50[i-1]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (aggressive when 1d bullish) ===
        if bull_regime:
            # Primary: EMA crossover with RSI confirmation
            if ema_cross_long and rsi_ok_long:
                new_signal = SIZE_LONG
            
            # Secondary: Pullback to EMA21 in uptrend
            elif bull_trend_12h and close[i] <= ema_21[i] * 1.01 and close[i] >= ema_21[i] * 0.99 and rsi[i] > 40:
                new_signal = SIZE_LONG
            
            # Tertiary: Keltner breakout with trend
            elif keltner_breakout_long and bull_trend_12h and above_200:
                new_signal = SIZE_LONG
            
            # Momentum: Higher low with bullish setup
            elif higher_low and bull_trend_12h and rsi[i] > 45:
                new_signal = SIZE_LONG
        
        # === SHORT ENTRIES (conservative when 1d bearish) ===
        elif bear_regime:
            # Primary: EMA crossover with RSI confirmation
            if ema_cross_short and rsi_ok_short:
                new_signal = -SIZE_SHORT
            
            # Secondary: Bounce to EMA21 in downtrend
            elif bear_trend_12h and close[i] >= ema_21[i] * 0.99 and close[i] <= ema_21[i] * 1.01 and rsi[i] < 60:
                new_signal = -SIZE_SHORT
            
            # Tertiary: Keltner breakdown with trend
            elif keltner_breakout_short and bear_trend_12h and below_200:
                new_signal = -SIZE_SHORT
            
            # Momentum: Lower high with bearish setup
            elif lower_high and bear_trend_12h and rsi[i] < 55:
                new_signal = -SIZE_SHORT
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss - trailing
        if position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss - trailing
        if position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            # New entry
            entry_price = close[i]
            position_side = np.sign(new_signal)
            if position_side > 0:
                trailing_stop = close[i] - 2.5 * atr[i]
                highest_close = close[i]
                lowest_close = 0.0
            else:
                trailing_stop = close[i] + 2.5 * atr[i]
                lowest_close = close[i]
                highest_close = 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            # Reversal
            entry_price = close[i]
            position_side = np.sign(new_signal)
            if position_side > 0:
                trailing_stop = close[i] - 2.5 * atr[i]
                highest_close = close[i]
                lowest_close = 0.0
            else:
                trailing_stop = close[i] + 2.5 * atr[i]
                lowest_close = close[i]
                highest_close = 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            # Exit
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals