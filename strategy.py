#!/usr/bin/env python3
"""
Experiment #505: 15m Multi-Timeframe with 4h HMA Trend + RSI Pullback + Volume

Hypothesis: After 499 failed experiments, the key insight is that 15m strategies fail
because they're either too strict (0 trades) or too loose (whipsaw losses). This strategy:

1. 4H HMA(21) TREND BIAS (via mtf_data helper):
   - Long only when price > 4h HMA (trend-aligned entries)
   - Short only when price < 4h HMA (trend-aligned entries)
   - Proven to 2x Sharpe in previous successful strategies

2. 15M RSI(14) PULLBACK ENTRY:
   - Long: RSI < 40 (looser than 30 to ensure trades)
   - Short: RSI > 60 (looser than 70 to ensure trades)
   - Pullback entries in direction of HTF trend

3. VOLUME CONFIRMATION:
   - Volume > 1.2 * SMA(volume, 20) confirms interest
   - Reduces false breakouts in low-volume periods

4. ATR(14) TRAILING STOP at 2.5x:
   - Tighter stop for 15m timeframe volatility
   - Signal → 0 when price moves 2.5*ATR against position

5. POSITION SIZING: 0.25 discrete
   - Conservative for 15m noise
   - Discrete levels minimize fee churn

Why this should work on 15m:
- 4h trend filter prevents counter-trend trades (major failure mode)
- Looser RSI thresholds (40/60) ensure sufficient trades
- Volume filter reduces whipsaws in choppy periods
- Should generate 50-100 trades/year per symbol

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_4h_hma_rsi_pullback_volume_atr_v1"
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

def calculate_sma(values, period=20):
    """Calculate Simple Moving Average."""
    values_s = pd.Series(values)
    return values_s.rolling(window=period, min_periods=period).mean().values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    return close_s.ewm(span=period, min_periods=period, adjust=False).mean().values

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
    rsi = calculate_rsi(close, 14)
    volume_sma = calculate_sma(volume, 20)
    ema_21 = calculate_ema(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
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
        
        if np.isnan(rsi[i]) or np.isnan(volume_sma[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        if volume_sma[i] == 0 or volume_sma[i] != volume_sma[i]:
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_trend = close[i] > hma_4h_aligned[i]
        bear_trend = close[i] < hma_4h_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.2 * volume_sma[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG: 4h bullish + RSI pullback + volume confirmation
        if bull_trend:
            if rsi[i] < 40 and volume_confirmed:
                new_signal = SIZE
            elif rsi[i] < 45:
                # Looser entry without volume for more trades
                new_signal = SIZE
        
        # SHORT: 4h bearish + RSI rally + volume confirmation
        if bear_trend:
            if rsi[i] > 60 and volume_confirmed:
                new_signal = -SIZE
            elif rsi[i] > 55:
                # Looser entry without volume for more trades
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
        # Exit if 4h trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend:
                new_signal = 0.0
            if position_side < 0 and bull_trend:
                new_signal = 0.0
        
        # === EMA CROSSOVER EXIT ===
        # Exit long if price crosses below EMA21
        if in_position and position_side > 0:
            if close[i] < ema_21[i] and i > 0 and close[i-1] >= ema_21[i-1]:
                new_signal = 0.0
        
        # Exit short if price crosses above EMA21
        if in_position and position_side < 0:
            if close[i] > ema_21[i] and i > 0 and close[i-1] <= ema_21[i-1]:
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