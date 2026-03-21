#!/usr/bin/env python3
"""
Experiment #346: 4h ADX Trend + Daily HMA + RSI Entry + Volume Confirm + ATR Stop
Hypothesis: 4h timeframe captures medium-term trends well. Using ADX(14)>20 for trend
strength (not too strict like ADX>40), Daily HMA(21) for macro bias, RSI(14) 45-55
range for entries (loose enough for trades), and volume>vol_sma20 for confirmation.
This avoids over-filtering that caused 0 trades in recent experiments while maintaining
signal quality. ATR(14)*2.5 trailing stop controls drawdown. Position size=0.25.
Timeframe: 4h (REQUIRED), HTF: 1d for trend bias via mtf_data helper.
Target: Beat Sharpe=0.499 with 30-60 trades/year, controlled DD<-30%.
Key insight: ADX>20 is achievable frequently vs ADX>40, RSI 45-55 allows more entries
than extreme values, volume filter removes low-liquidity false signals.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_adx_trend_daily_hma_rsi_volume_atr_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    # Calculate +DM and -DM
    plus_dm = np.where(high - np.roll(high, 1) > np.roll(low, 1) - low, 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where(np.roll(low, 1) - low > high - np.roll(high, 1),
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Calculate ATR for TR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Smooth TR, +DM, -DM using Wilder's method (EMA with span=period)
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
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
    
    for i in range(250, n):  # Start after 250 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Daily macro trend bias
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # ADX trend strength (loose threshold >20, not >40)
        trend_strong = adx[i] > 20
        
        # DI crossover for direction
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = minus_di[i] > plus_di[i]
        
        # RSI entry zone (loose - not extremes)
        rsi_ok_long = rsi[i] > 45 and rsi[i] < 70  # Not overbought
        rsi_ok_short = rsi[i] < 55 and rsi[i] > 30  # Not oversold
        
        # Volume confirmation (above average)
        volume_ok = volume[i] > vol_sma[i]
        
        # Price momentum (close > open for long, close < open for short)
        price_momentum_long = close[i] > prices["open"].values[i]
        price_momentum_short = close[i] < prices["open"].values[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: Daily bullish + ADX strong + DI bullish + RSI ok + Volume ok
        if daily_bullish and trend_strong and di_bullish and rsi_ok_long and volume_ok:
            new_signal = SIZE_ENTRY
        # Secondary: Daily bullish + DI bullish + Rsi ok (no ADX filter for more trades)
        elif daily_bullish and di_bullish and rsi_ok_long and price_momentum_long:
            new_signal = SIZE_ENTRY
        # Tertiary: Breakout momentum (RSI strong + volume)
        elif rsi[i] > 55 and rsi[i] < 70 and volume_ok and price_momentum_long:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: Daily bearish + ADX strong + DI bearish + RSI ok + Volume ok
        if daily_bearish and trend_strong and di_bearish and rsi_ok_short and volume_ok:
            new_signal = -SIZE_ENTRY
        # Secondary: Daily bearish + DI bearish + RSI ok (no ADX filter for more trades)
        elif daily_bearish and di_bearish and rsi_ok_short and price_momentum_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: Breakdown momentum (RSI weak + volume)
        elif rsi[i] < 45 and rsi[i] > 30 and volume_ok and price_momentum_short:
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