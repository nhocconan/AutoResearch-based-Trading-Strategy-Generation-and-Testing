#!/usr/bin/env python3
"""
Experiment #305: 12h HMA Trend + Daily Macro Bias + RSI Mean Reversion with ATR Stops
Hypothesis: 12h timeframe balances trend capture with trade frequency. Daily HMA provides 
macro trend bias while 12h RSI pullbacks (30-50 long, 50-70 short) ensure entries on dips/rallies.
Simpler entry logic than #294 to guarantee >=10 trades per symbol (learned from 0-trade failures).
ATR trailing stops (2.5*ATR) control drawdown. Position size 0.30 balances returns vs risk.
Target: Beat Sharpe=0.499 from current best while ensuring >=10 trades per symbol on 12h timeframe.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_daily_bias_rsi_mean_reversion_atr_v1"
timeframe = "12h"
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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average for long-term trend filter."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average for volume spike detection."""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    hma_12h_21 = calculate_hma(close, 21)
    hma_12h_50 = calculate_hma(close, 50)
    rsi = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    vol_ma = calculate_volume_ma(volume, 20)
    
    # Track previous values for crossover detection
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_hma_21 = np.roll(hma_12h_21, 1)
    prev_hma_21[0] = hma_12h_21[0]
    prev_rsi = np.roll(rsi, 1)
    prev_rsi[0] = rsi[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_12h_21[i]) or np.isnan(hma_12h_50[i]) or np.isnan(sma_200[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        # Daily macro trend bias
        daily_bullish = close[i] > hma_1d_21_aligned[i] and hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i] and hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # 12h trend filter
        trend_bullish = close[i] > hma_12h_21[i] and hma_12h_21[i] > hma_12h_50[i]
        trend_bearish = close[i] < hma_12h_21[i] and hma_12h_21[i] < hma_12h_50[i]
        
        # Long-term trend filter (SMA200)
        above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else False
        below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # RSI mean reversion zones (generous ranges to ensure trades)
        rsi_oversold = rsi[i] < 45
        rsi_overbought = rsi[i] > 55
        rsi_extreme_low = rsi[i] < 35
        rsi_extreme_high = rsi[i] > 65
        rsi_neutral = 40 < rsi[i] < 60
        
        # Volume confirmation
        volume_spike = volume[i] > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        # HMA crossover signals
        hma_cross_long = prev_close[i] <= prev_hma_21[i] and close[i] > hma_12h_21[i]
        hma_cross_short = prev_close[i] >= prev_hma_21[i] and close[i] < hma_12h_21[i]
        
        # HMA slope (trend direction)
        hma_slope_bullish = hma_12h_21[i] > prev_hma_21[i]
        hma_slope_bearish = hma_12h_21[i] < prev_hma_21[i]
        
        # RSI crossover for entry timing
        rsi_cross_up = prev_rsi[i] < 40 and rsi[i] >= 40
        rsi_cross_down = prev_rsi[i] > 60 and rsi[i] <= 60
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: Daily bullish + 12h trend + RSI oversold + HMA cross
        if daily_bullish and trend_bullish and rsi_oversold and hma_cross_long:
            new_signal = SIZE_ENTRY
        # Secondary: Daily bullish + Above SMA200 + RSI cross up + Price > HMA
        elif daily_bullish and above_sma200 and rsi_cross_up and close[i] > hma_12h_21[i]:
            new_signal = SIZE_ENTRY
        # Tertiary: 12h trend + RSI extreme low + HMA slope bullish (simpler for more trades)
        elif trend_bullish and rsi_extreme_low and hma_slope_bullish:
            new_signal = SIZE_ENTRY
        # Quaternary: Price > HMA21 > HMA50 + RSI neutral + Volume spike
        elif close[i] > hma_12h_21[i] > hma_12h_50[i] and rsi_neutral and volume_spike:
            new_signal = SIZE_ENTRY
        # Simple: Daily bullish + Price > HMA21 + RSI > 40 (most permissive)
        elif daily_bullish and close[i] > hma_12h_21[i] and rsi[i] > 40 and rsi[i] < 70:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Primary: Daily bearish + 12h trend + RSI overbought + HMA cross
        if daily_bearish and trend_bearish and rsi_overbought and hma_cross_short:
            new_signal = -SIZE_ENTRY
        # Secondary: Daily bearish + Below SMA200 + RSI cross down + Price < HMA
        elif daily_bearish and below_sma200 and rsi_cross_down and close[i] < hma_12h_21[i]:
            new_signal = -SIZE_ENTRY
        # Tertiary: 12h trend + RSI extreme high + HMA slope bearish (simpler for more trades)
        elif trend_bearish and rsi_extreme_high and hma_slope_bearish:
            new_signal = -SIZE_ENTRY
        # Quaternary: Price < HMA21 < HMA50 + RSI neutral + Volume spike
        elif close[i] < hma_12h_21[i] < hma_12h_50[i] and rsi_neutral and volume_spike:
            new_signal = -SIZE_ENTRY
        # Simple: Daily bearish + Price < HMA21 + RSI < 60 (most permissive)
        elif daily_bearish and close[i] < hma_12h_21[i] and rsi[i] < 60 and rsi[i] > 30:
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