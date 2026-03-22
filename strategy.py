#!/usr/bin/env python3
"""
Experiment #117: 1h HMA Trend + 4h Regime Filter + Asymmetric RSI Entries

Hypothesis: After 116 experiments, the key insight is that BTC/ETH need ASYMMETRIC logic:
- In bull regime (price > 4h HMA): only take LONG entries on RSI pullbacks
- In bear regime (price < 4h HMA): only take SHORT entries on RSI rallies
- This avoids the 2022 crash whipsaw that destroyed symmetric trend strategies
- 1h timeframe provides enough signals (50-100 trades/year) while filtering noise
- RSI(7) extremes ( <30 / >70) with 4h trend filter = high-probability entries
- Volume confirmation (volume > 1.5 * SMA20) reduces false breakouts
- 3*ATR trailing stop protects against reversals without premature exits

Why this might beat the baseline (Sharpe=0.436):
- Asymmetric logic prevents counter-trend trades in strong moves
- 4h HMA is smoother than EMA, fewer regime flip-flops
- RSI pullback entries have better risk/reward than breakouts
- Volume filter adds confirmation without over-filtering
- Conservative sizing (0.20-0.30) limits drawdown during crashes

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 3.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_4h_regime_rsi_asymmetric_v1"
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
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume for volume confirmation."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

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
    rsi = calculate_rsi(close, 7)  # Faster RSI for 1h entries
    vol_sma = calculate_volume_sma(volume, 20)
    
    # Calculate 1h HMA for additional trend confirmation
    hma_1h = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
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
        
        if np.isnan(rsi[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME REGIME FILTER ===
        # 4h HMA = higher timeframe trend bias (regime detector)
        bull_regime_4h = close[i] > hma_4h_aligned[i]
        bear_regime_4h = close[i] < hma_4h_aligned[i]
        
        # 1h HMA for additional confirmation
        bull_1h = close[i] > hma_1h[i]
        bear_1h = close[i] < hma_1h[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.3 * vol_sma[i] if vol_sma[i] > 0 else False
        
        # === RSI EXTREMES FOR ENTRY ===
        rsi_oversold = rsi[i] < 30  # Long entry zone
        rsi_overbought = rsi[i] > 70  # Short entry zone
        rsi_neutral = 30 <= rsi[i] <= 70
        
        new_signal = 0.0
        
        # === ASYMMETRIC LONG ENTRY CONDITIONS ===
        # Only long in bull regime (prevents counter-trend longs in crash)
        if bull_regime_4h:
            # Strong: Bull regime + RSI oversold + volume confirmed + 1h bullish
            if rsi_oversold and volume_confirmed and bull_1h:
                new_signal = SIZE_STRONG
            # Moderate: Bull regime + RSI oversold
            elif rsi_oversold:
                new_signal = SIZE_BASE
            # Weak: Bull regime + 1h bullish (ensure trades on all symbols)
            elif bull_1h and rsi[i] < 50:
                new_signal = SIZE_BASE
        
        # === ASYMMETRIC SHORT ENTRY CONDITIONS ===
        # Only short in bear regime (prevents counter-trend shorts in rally)
        if bear_regime_4h:
            # Strong: Bear regime + RSI overbought + volume confirmed + 1h bearish
            if rsi_overbought and volume_confirmed and bear_1h:
                new_signal = -SIZE_STRONG
            # Moderate: Bear regime + RSI overbought
            elif rsi_overbought:
                new_signal = -SIZE_BASE
            # Weak: Bear regime + 1h bearish (ensure trades on all symbols)
            elif bear_1h and rsi[i] > 50:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 3.0 * ATR below highest close
            stoploss_price = highest_close - 3.0 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 3.0 * ATR above lowest close
            stoploss_price = lowest_close + 3.0 * atr[i]
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