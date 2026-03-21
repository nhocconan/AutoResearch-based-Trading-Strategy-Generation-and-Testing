#!/usr/bin/env python3
"""
Experiment #373: 15m Donchian Breakout + 4h HMA Trend + 1h RSI Momentum + Volume Filter + ATR Stop
Hypothesis: 15m timeframe captures intraday breakouts while 4h HMA provides trend bias and 1h RSI
filters momentum. Donchian breakout (20-period) catches volatility expansions. Volume confirmation
(>1.5x 20-bar SMA) reduces false breakouts. This should generate MORE trades than 12h strategies
while maintaining quality through multi-timeframe filters. ATR(14) stoploss at 2.5x protects capital.
Timeframe: 15m (REQUIRED), HTF: 4h for trend, 1h for momentum via mtf_data helper.
Target: Beat Sharpe=0.499 with 50-150 trades total, DD < -30%.
Key insight: 15m breakouts with HTF confirmation should work in both trending and ranging markets.
Build on #371 by moving to faster timeframe with Donchian breakout instead of KAMA crossover.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_donchian_4h_hma_1h_rsi_volume_atr_v1"
timeframe = "15m"
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
    """Calculate Hull Moving Average for faster trend response."""
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

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    upper[:period] = np.nan
    lower[:period] = np.nan
    return upper, lower

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    vol_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # 4h trend bias
        trend_bullish = not np.isnan(hma_4h_aligned[i]) and close[i] > hma_4h_aligned[i]
        trend_bearish = not np.isnan(hma_4h_aligned[i]) and close[i] < hma_4h_aligned[i]
        
        # 1h RSI momentum filter
        rsi_ok_long = not np.isnan(rsi_1h_aligned[i]) and rsi_1h_aligned[i] > 40 and rsi_1h_aligned[i] < 75
        rsi_ok_short = not np.isnan(rsi_1h_aligned[i]) and rsi_1h_aligned[i] > 25 and rsi_1h_aligned[i] < 60
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_sma[i]
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i-1] and close[i-1] <= donchian_upper[i-1]
        breakout_short = close[i] < donchian_lower[i-1] and close[i-1] >= donchian_lower[i-1]
        
        # Donchian position (already broken out)
        donchian_bullish = close[i] > (donchian_upper[i] + donchian_lower[i]) / 2
        donchian_bearish = close[i] < (donchian_upper[i] + donchian_lower[i]) / 2
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: Donchian breakout long + 4h bullish + 1h RSI ok + Volume confirmed
        if breakout_long and trend_bullish and rsi_ok_long and volume_confirmed:
            new_signal = SIZE_ENTRY
        # Secondary: Donchian bullish + 4h bullish + 1h RSI ok (no breakout needed)
        elif donchian_bullish and trend_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Tertiary: Donchian breakout long + Volume confirmed (4h neutral ok)
        elif breakout_long and volume_confirmed and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Quaternary: Donchian bullish alone with volume (ensures minimum trade frequency)
        elif donchian_bullish and volume[i] > 1.2 * vol_sma[i] and rsi_ok_long:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: Donchian breakout short + 4h bearish + 1h RSI ok + Volume confirmed
        if breakout_short and trend_bearish and rsi_ok_short and volume_confirmed:
            new_signal = -SIZE_ENTRY
        # Secondary: Donchian bearish + 4h bearish + 1h RSI ok (no breakout needed)
        elif donchian_bearish and trend_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: Donchian breakout short + Volume confirmed (4h neutral ok)
        elif breakout_short and volume_confirmed and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Quaternary: Donchian bearish alone with volume (ensures minimum trade frequency)
        elif donchian_bearish and volume[i] > 1.2 * vol_sma[i] and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
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