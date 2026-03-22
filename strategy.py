#!/usr/bin/env python3
"""
Experiment #009: 1h Donchian Breakout + 4h HMA Trend + Volume Z-Score + RSI Filter + ATR Stop
Hypothesis: 1h timeframe balances trade frequency and noise reduction. Donchian breakouts 
capture momentum moves, 4h HMA provides HTF trend bias to avoid counter-trend trades,
volume z-score confirms breakout validity (avoids fakeouts), RSI filters extreme entries.
Multiple entry paths ensure >=10 trades per symbol. Conservative sizing (0.25) controls DD.
2.0*ATR stoploss appropriate for 1h bars. Must beat Sharpe=0.121 baseline.
Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_donchian_4h_hma_vol_zscore_rsi_atr_v1"
timeframe = "1h"
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
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_volume_zscore(volume, period=20):
    """Calculate Z-score of volume for volume confirmation."""
    vol_s = pd.Series(volume)
    vol_mean = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_std = vol_s.rolling(window=period, min_periods=period).std().values
    zscore = np.where(vol_std > 0, (volume - vol_mean) / vol_std, 0.0)
    return zscore

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    for i in range(er_period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        if i == er_period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    vol_zscore = calculate_volume_zscore(volume, 20)
    kama = calculate_kama(close, 10, 2, 30)
    
    # Additional 1h HMA for trend confirmation
    hma_1h = calculate_hma(close, 21)
    hma_1h_fast = calculate_hma(close, 10)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(vol_zscore[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(hma_1h[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        htf_bullish = close[i] > hma_4h_aligned[i]
        htf_bearish = close[i] < hma_4h_aligned[i]
        
        # 1h trend confirmation
        ltf_bullish = close[i] > hma_1h[i]
        ltf_bearish = close[i] < hma_1h[i]
        hma_rising = hma_1h[i] > hma_1h[i - 1] if i > 0 else False
        hma_falling = hma_1h[i] < hma_1h[i - 1] if i > 0 else False
        
        # Fast HMA crossover
        fast_above_slow = hma_1h_fast[i] > hma_1h[i]
        fast_below_slow = hma_1h_fast[i] < hma_1h[i]
        
        # KAMA trend
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # Donchian breakout signals
        donchian_breakout_long = close[i] > donchian_upper[i - 1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i - 1] if i > 0 else False
        
        # Volume confirmation
        vol_confirmed = vol_zscore[i] > 0.5  # Above average volume
        
        # RSI zones
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = rsi[i] > 40 and rsi[i] < 60
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: Donchian breakout + HTF bullish + Volume confirmed + RSI not overbought
        if donchian_breakout_long and htf_bullish and vol_confirmed and rsi[i] < 70:
            new_signal = SIZE_ENTRY
        
        # Path 2: HTF bullish + LTF bullish + Fast HMA crossover + RSI bullish
        elif htf_bullish and ltf_bullish and fast_above_slow and rsi_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 3: HTF bullish + KAMA bullish + RSI pullback (40-50) + Volume ok
        elif htf_bullish and kama_bullish and rsi[i] > 40 and rsi[i] < 50 and vol_zscore[i] > -0.5:
            new_signal = SIZE_ENTRY
        
        # Path 4: HTF bullish + HMA rising + RSI neutral + Donchian near upper
        elif htf_bullish and hma_rising and rsi_neutral and close[i] > donchian_upper[i] * 0.98 if not np.isnan(donchian_upper[i]) else False:
            new_signal = SIZE_ENTRY
        
        # Path 5: HTF bullish + RSI oversold bounce (mean reversion in uptrend)
        elif htf_bullish and rsi_oversold and rsi[i] > rsi[i - 1] if i > 0 else False:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: Donchian breakdown + HTF bearish + Volume confirmed + RSI not oversold
        if donchian_breakout_short and htf_bearish and vol_confirmed and rsi[i] > 30:
            new_signal = -SIZE_ENTRY
        
        # Path 2: HTF bearish + LTF bearish + Fast HMA crossover down + RSI bearish
        elif htf_bearish and ltf_bearish and fast_below_slow and rsi_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 3: HTF bearish + KAMA bearish + RSI pullback (50-60) + Volume ok
        elif htf_bearish and kama_bearish and rsi[i] > 50 and rsi[i] < 60 and vol_zscore[i] > -0.5:
            new_signal = -SIZE_ENTRY
        
        # Path 4: HTF bearish + HMA falling + RSI neutral + Donchian near lower
        elif htf_bearish and hma_falling and rsi_neutral and close[i] < donchian_lower[i] * 1.02 if not np.isnan(donchian_lower[i]) else False:
            new_signal = -SIZE_ENTRY
        
        # Path 5: HTF bearish + RSI overbought drop (mean reversion in downtrend)
        elif htf_bearish and rsi_overbought and rsi[i] < rsi[i - 1] if i > 0 else False:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR for 1h timeframe)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR for 1h timeframe)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals