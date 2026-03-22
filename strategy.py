#!/usr/bin/env python3
"""
Experiment #241: 15m Multi-Timeframe Trend Pullback with Volume Confirmation

Hypothesis: 15m timeframe is noisy, but combining strong HTF trend filter (4h HMA)
with pullback entries (1h RSI) and volume confirmation can capture trend continuations
while avoiding false breakouts. Key insights from failed experiments:
- Pure trend following fails in 2025 bear/range market
- Lower TF (15m/30m) strategies had Sharpe -3 to -9 due to noise
- Need STRONG HTF bias to filter 15m noise
- Volume confirmation reduces false signals

Strategy logic:
1. 4h HMA(21) = primary trend bias (only trade in trend direction)
2. 1h RSI(14) pullback = entry trigger (RSI < 45 for long, RSI > 55 for short)
3. Volume > 1.3x 20-bar avg = confirmation (avoids low-volume false breaks)
4. ATR(14) trailing stop = 2.5x ATR protection
5. Discrete sizing: 0.20 base, 0.25 strong conviction

Why 15m might work now:
- Previous 15m failures lacked strong HTF filter
- Adding 4h HMA + 1h RSI creates multi-layer confirmation
- Volume filter avoids whipsaw on low-liquidity bars
- Conservative sizing (0.20-0.25) controls drawdown

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h HMA + 1h RSI via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_trend_pullback_4h_hma_1h_rsi_volume_atr_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_15m = calculate_rsi(close, 14)
    vol_ma = calculate_volume_ma(volume, 20)
    
    # Calculate 15m EMA for additional trend confirmation
    close_s = pd.Series(close)
    ema_21 = close_s.ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    entry_idx = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_15m[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME TREND BIAS (4h HMA) ===
        # Only trade in direction of 4h trend
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === 1h RSI PULLBACK FILTER ===
        # In uptrend: wait for RSI pullback to 40-50 zone
        # In downtrend: wait for RSI rally to 50-60 zone
        rsi_1h_neutral = 45 < rsi_1h_aligned[i] < 55
        rsi_1h_bullish_pullback = 40 < rsi_1h_aligned[i] < 50
        rsi_1h_bearish_rally = 50 < rsi_1h_aligned[i] < 60
        
        # === 15m VOLUME CONFIRMATION ===
        # Volume must be above average to confirm move
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # === 15m TREND STRUCTURE ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === 15m RSI MOMENTUM ===
        rsi_15m_oversold = rsi_15m[i] < 45
        rsi_15m_overbought = rsi_15m[i] > 55
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # 4h bullish + 1h RSI pullback + 15m volume + 15m EMA bullish + 15m RSI confirmation
        if bull_trend_4h and rsi_1h_bullish_pullback and volume_confirmed and ema_bullish and rsi_15m_oversold:
            new_signal = SIZE_BASE
        
        # Strong long: all conditions + 1h RSI very bullish
        if bull_trend_4h and rsi_1h_aligned[i] < 45 and volume_confirmed and ema_bullish:
            new_signal = SIZE_STRONG
        
        # === SHORT ENTRY ===
        # 4h bearish + 1h RSI rally + 15m volume + 15m EMA bearish + 15m RSI confirmation
        if bear_trend_4h and rsi_1h_bearish_rally and volume_confirmed and ema_bearish and rsi_15m_overbought:
            new_signal = -SIZE_BASE
        
        # Strong short: all conditions + 1h RSI very bearish
        if bear_trend_4h and rsi_1h_aligned[i] > 55 and volume_confirmed and ema_bearish:
            new_signal = -SIZE_STRONG
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TAKE PROFIT: Reduce to half at 2R profit ===
        if in_position and new_signal != 0.0 and position_side != 0:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.0 * atr[entry_idx]:
                    new_signal = SIZE_BASE / 2  # Take partial profit
            if position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.0 * atr[entry_idx]:
                    new_signal = -SIZE_BASE / 2  # Take partial profit
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_idx = i
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_idx = i
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            # Exiting position (signal-based or stoploss)
            if in_position and signals[i-1] != 0.0:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals