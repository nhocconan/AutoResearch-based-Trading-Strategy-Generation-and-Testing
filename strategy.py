#!/usr/bin/env python3
"""
Experiment #295: 15m Supertrend + 4h/1h HMA Trend Filter + RSI Momentum
Hypothesis: 15m timeframe captures intraday momentum while 4h/1h HMA provides trend bias.
Supertrend(10,3) gives clear entry/exit signals. RSI(14) filter avoids extreme entries.
Volume confirmation ensures breakout validity. ATR(14) trailing stop controls drawdown.
Position size 0.25-0.30 with discrete levels minimizes fee churn while generating >=10 trades.
Target: Beat Sharpe=0.499 from current best while ensuring trades on ALL symbols (BTC/ETH/SOL).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_supertrend_4h_1h_hma_rsi_volume_atr_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2.0
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    n = len(close)
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = bullish, -1 = bearish
    
    final_upper[0] = upper_band[0]
    final_lower[0] = lower_band[0]
    supertrend[0] = lower_band[0]
    
    for i in range(1, n):
        # Update upper/lower bands
        if upper_band[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = upper_band[i]
        else:
            final_upper[i] = final_upper[i-1]
        
        if lower_band[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = lower_band[i]
        else:
            final_lower[i] = final_lower[i-1]
        
        # Determine supertrend direction
        if direction[i-1] == 1:
            if close[i] < final_lower[i]:
                direction[i] = -1
                supertrend[i] = final_upper[i]
            else:
                direction[i] = 1
                supertrend[i] = final_lower[i]
        else:
            if close[i] > final_upper[i]:
                direction[i] = 1
                supertrend[i] = final_lower[i]
            else:
                direction[i] = -1
                supertrend[i] = final_upper[i]
    
    return supertrend, direction

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

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    return vol_ratio

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
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, 50)
    hma_1h_21 = calculate_hma(df_1h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    hma_1h_21_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_21)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    rsi = calculate_rsi(close, 14)
    sma_50 = calculate_sma(close, 50)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Track previous values for crossover detection
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_st_direction = np.roll(st_direction, 1)
    prev_st_direction[0] = st_direction[0]
    prev_rsi = np.roll(rsi, 1)
    prev_rsi[0] = rsi[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28
    SIZE_HALF = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(supertrend[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            continue
        
        # 4h macro trend bias
        trend_4h_bullish = close[i] > hma_4h_21_aligned[i] and hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_21_aligned[i] and hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # 1h intermediate trend
        trend_1h_bullish = close[i] > hma_1h_21_aligned[i]
        trend_1h_bearish = close[i] < hma_1h_21_aligned[i]
        
        # 15m Supertrend direction
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # Supertrend crossover signals
        st_cross_long = prev_st_direction[i] == -1 and st_direction[i] == 1
        st_cross_short = prev_st_direction[i] == 1 and st_direction[i] == -1
        
        # RSI filter (avoid extremes)
        rsi_ok_long = 35 < rsi[i] < 75
        rsi_ok_short = 25 < rsi[i] < 65
        rsi_momentum_long = rsi[i] > prev_rsi[i] and rsi[i] > 45
        rsi_momentum_short = rsi[i] < prev_rsi[i] and rsi[i] < 55
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 1.0
        
        # Price above/below SMA50
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: 4h bullish + 1h bullish + Supertrend cross + RSI momentum + Volume
        if trend_4h_bullish and trend_1h_bullish and st_cross_long and rsi_momentum_long and vol_confirmed:
            new_signal = SIZE_ENTRY
        # Secondary: 4h bullish + Supertrend bullish + RSI ok + Price > SMA50
        elif trend_4h_bullish and st_bullish and rsi_ok_long and above_sma50:
            new_signal = SIZE_ENTRY
        # Tertiary: 1h bullish + Supertrend cross + RSI momentum (simpler for more trades)
        elif trend_1h_bullish and st_cross_long and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        # Quaternary: Supertrend cross + RSI 40-60 + Volume (momentum entry)
        elif st_cross_long and 40 < rsi[i] < 60 and vol_confirmed:
            new_signal = SIZE_ENTRY
        # Simple: 4h bullish + Supertrend bullish + RSI > 45 (trend continuation)
        elif trend_4h_bullish and st_bullish and rsi[i] > 45 and rsi_ok_long:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Primary: 4h bearish + 1h bearish + Supertrend cross + RSI momentum + Volume
        if trend_4h_bearish and trend_1h_bearish and st_cross_short and rsi_momentum_short and vol_confirmed:
            new_signal = -SIZE_ENTRY
        # Secondary: 4h bearish + Supertrend bearish + RSI ok + Price < SMA50
        elif trend_4h_bearish and st_bearish and rsi_ok_short and below_sma50:
            new_signal = -SIZE_ENTRY
        # Tertiary: 1h bearish + Supertrend cross + RSI momentum (simpler for more trades)
        elif trend_1h_bearish and st_cross_short and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        # Quaternary: Supertrend cross + RSI 40-60 + Volume (momentum entry)
        elif st_cross_short and 40 < rsi[i] < 60 and vol_confirmed:
            new_signal = -SIZE_ENTRY
        # Simple: 4h bearish + Supertrend bearish + RSI < 55 (trend continuation)
        elif trend_4h_bearish and st_bearish and rsi[i] < 55 and rsi_ok_short:
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