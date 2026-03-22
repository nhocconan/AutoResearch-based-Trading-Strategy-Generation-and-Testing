#!/usr/bin/env python3
"""
Experiment #169: 15m Vol Spike Mean Reversion + 4h HMA Trend Filter + RSI Timing

Hypothesis: 15m timeframe captures short-term volatility spikes that quickly revert.
When ATR(7)/ATR(30) > 2.0, volatility is abnormally high and likely to compress.
Combined with Bollinger Band extremes (price < BB lower) and RSI oversold, this
creates high-probability mean reversion entries. 4h HMA provides trend bias to
avoid counter-trend trades that fail in strong trends.

Why 15m might work for vol spike reversion:
- Volatility clusters on 15m during panic/euphoria moments
- Mean reversion faster on 15m than higher timeframes
- 4h HTF filter prevents trading against major trend
- ATR ratio filter ensures we only enter during extreme vol events

Learning from failures:
- #157, #163 (15m CRSI): Negative Sharpe - pure mean reversion fails without vol filter
- #158, #159 (30m/1h EMA): Negative Sharpe - trend following alone doesn't work
- Need combination: vol spike + mean reversion + HTF trend bias
- Must ensure ≥10 trades per symbol (don't over-filter with too many conditions)

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_vol_spike_4h_hma_rsi_bb_atr_v1"
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
    """Calculate RSI using Wilder's smoothing."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    # Volatility ratio: ATR(7) / ATR(30)
    vol_ratio = atr_7 / (atr_30 + 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(bb_lower[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === VOLATILITY SPIKE FILTER ===
        # ATR(7)/ATR(30) > 2.0 = volatility is abnormally high (panic/euphoria)
        vol_spike = vol_ratio[i] > 2.0
        
        # === BOLLINGER BAND EXTREMES ===
        # Price below lower BB = oversold extreme
        # Price above upper BB = overbought extreme
        bb_oversold = close[i] < bb_lower[i]
        bb_overbought = close[i] > bb_upper[i]
        
        # === RSI EXTREMES ===
        # RSI < 30 = oversold
        # RSI > 70 = overbought
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # 4h bullish trend + vol spike + BB oversold + RSI oversold
        # Asymmetric: only long when 4h trend is bullish
        if bull_trend_4h and vol_spike and bb_oversold and rsi_oversold:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # 4h bearish trend + vol spike + BB overbought + RSI overbought
        # Asymmetric: only short when 4h trend is bearish
        if bear_trend_4h and vol_spike and bb_overbought and rsi_overbought:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr_14[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr_14[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals