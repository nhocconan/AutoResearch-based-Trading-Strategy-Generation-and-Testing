#!/usr/bin/env python3
"""
Experiment #507: 1h Multi-Timeframe Mean Reversion with 4h HMA Trend Filter

Hypothesis: After 506 failed experiments, the pattern is clear - complex regime 
detection and adaptive logic have failed repeatedly. The winning approach is 
SIMPLE: 4h HMA for robust trend bias + 1h RSI for quick mean-reversion entries.
Key insight from current best (Sharpe=0.676): multi-timeframe alignment is critical,
but entry logic should be simple and loose to ensure sufficient trades.

Strategy Design:
1. 4h HMA(21) trend bias via mtf_data helper (proven edge)
2. 1h RSI(14) for mean-reversion entries (30/70 thresholds - loose for trades)
3. ATR(14) * 2.5 trailing stop (tighter for 1h volatility)
4. Position sizing: 0.28 discrete (conservative for 1h noise)
5. Volume confirmation: taker_buy_volume ratio > 0.45 for longs, < 0.55 for shorts

Entry Logic:
- LONG: 4h HMA bullish + RSI(14) < 32 + volume confirmation
- SHORT: 4h HMA bearish + RSI(14) > 68 + volume confirmation

Why this should work on 1h:
- Simpler than failed experiments #495-#506 (all negative Sharpe)
- Looser RSI thresholds (32/68 vs 30/70) ensure 50-100 trades/year
- 4h HMA provides robust trend filter without whipsaw
- Volume confirmation reduces false signals
- Conservative sizing (0.28) limits drawdown on 1h noise

Timeframe: 1h (REQUIRED)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_volume_simple_atr_v1"
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

def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio."""
    ratio = taker_buy_volume / np.maximum(volume, 1e-10)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend bias
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align 4h HMA to 1h (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(taker_buy_volume, volume)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === 4h HMA TREND BIAS ===
        bull_trend = close[i] > hma_4h_aligned[i]
        bear_trend = close[i] < hma_4h_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_bullish = vol_ratio[i] > 0.45
        volume_bearish = vol_ratio[i] < 0.55
        
        # === ENTRY LOGIC (loose thresholds for sufficient trades) ===
        new_signal = 0.0
        
        # LONG: 4h bullish + RSI oversold + volume confirmation
        if bull_trend and rsi[i] < 32 and volume_bullish:
            new_signal = SIZE
        
        # SHORT: 4h bearish + RSI overbought + volume confirmation
        if bear_trend and rsi[i] > 68 and volume_bearish:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
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