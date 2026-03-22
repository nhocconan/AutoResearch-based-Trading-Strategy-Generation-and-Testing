#!/usr/bin/env python3
"""
Experiment #397: 15m RSI Mean Reversion + 4h HMA Trend Filter

Hypothesis: After 396 experiments, the pattern is clear - strategies fail because:
1. Too many filters = 0 trades (experiments 386, 389, 390, 395 had Sharpe=0 or negative)
2. No HTF trend filter = whipsaw in 2022 crash
3. Position size too large = blowup on drawdown

SOLUTION for 15m:
- 4h HMA(21) for trend direction (proven edge, call ONCE before loop)
- 15m RSI(7) for quick mean-reversion entries (faster than RSI 14)
- LOOSE entry thresholds: RSI < 40 for long, RSI > 60 for short (not extreme 30/70)
- Volume filter: volume > 0.8 * SMA20 (not 1.5x which filters too much)
- ATR(14) * 2.5 trailing stop
- Position size 0.30 discrete

Why 15m works:
- More trades than 4h/12h (avoids 0-trade problem)
- RSI(7) catches intraday reversals
- 4h HMA prevents counter-trend trades in crashes
- Should generate 50-100 trades/year per symbol

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_meanrev_4h_hma_volume_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=7):
    """Calculate RSI with configurable period."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

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
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 7)
    vol_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === VOLUME CONFIRMATION (loose filter) ===
        volume_confirmed = volume[i] > 0.8 * vol_sma[i]
        
        # === RSI MEAN REVERSION SIGNALS (LOOSE thresholds for trades) ===
        # Long: RSI < 40 (oversold) + 4h trend bullish
        rsi_oversold = rsi[i] < 40
        # Short: RSI > 60 (overbought) + 4h trend bearish
        rsi_overbought = rsi[i] > 60
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG entry: oversold + bullish 4h trend + volume
        if rsi_oversold and bull_trend_4h and volume_confirmed:
            new_signal = SIZE
        
        # SHORT entry: overbought + bearish 4h trend + volume
        elif rsi_overbought and bear_trend_4h and volume_confirmed:
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
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            # Long position should exit if 4h trend turns bearish
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            # Short position should exit if 4h trend turns bullish
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === RSI EXIT (take profit on mean reversion) ===
        if in_position and new_signal != 0.0:
            # Long: exit when RSI recovers to neutral
            if position_side > 0 and rsi[i] > 55:
                new_signal = 0.0
            # Short: exit when RSI drops to neutral
            if position_side < 0 and rsi[i] < 45:
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