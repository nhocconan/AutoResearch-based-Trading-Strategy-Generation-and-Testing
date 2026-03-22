#!/usr/bin/env python3
"""
Experiment #427: 15m EMA Pullback + 4h HMA Trend + Volume Confirmation + ATR Stop

Hypothesis: After analyzing 426 failed experiments, the key insight is that 15m
timeframe needs STRONG higher-timeframe filtering to avoid noise whipsaws.
This strategy uses a proven pullback approach adapted for 15m:

1. 4h HMA(21) TREND BIAS (via mtf_data helper):
   - Long only when 15m price > 4h HMA (bullish trend)
   - Short only when 15m price < 4h HMA (bearish trend)
   - HMA smoother than EMA, critical for MTF alignment

2. 15m EMA PULLBACK ENTRY:
   - Fast EMA(8) crosses above Slow EMA(21) for long (in uptrend)
   - Fast EMA(8) crosses below Slow EMA(21) for short (in downtrend)
   - Only enter on pullback, not breakout (reduces false signals)

3. VOLUME CONFIRMATION:
   - Volume must be > 1.3 * SMA(volume, 20) on entry bar
   - Confirms genuine interest, not noise

4. RSI(14) FILTER:
   - Long: RSI > 45 (not oversold, confirms momentum)
   - Short: RSI < 55 (not overbought, confirms momentum)
   - Avoids entering at extremes

5. ATR(14) TRAILING STOP at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Protects from adverse moves

6. POSITION SIZING: 0.25 discrete (conservative for 15m noise)
   - Max 25% capital per position
   - Discrete levels minimize fee churn

Why 15m should work with this approach:
- 4h HMA filter removes 15m noise and false breakouts
- Pullback entries (not breakouts) have better win rate
- Volume confirmation reduces false signals
- Should work on BTC/ETH/SOL individually (not SOL-biased)
- Generates 50-100 trades/year (enough for statistical significance)

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_ema_pullback_4h_hma_vol_rsi_atr_v1"
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

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

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
    ema_fast = calculate_ema(close, 8)
    ema_slow = calculate_ema(close, 21)
    rsi = calculate_rsi(close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    
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
        
        if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]):
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
        
        # === EMA PULLBACK SIGNALS ===
        # EMA crossover: fast crosses above slow = bullish momentum
        ema_bullish = ema_fast[i] > ema_slow[i]
        ema_bearish = ema_fast[i] < ema_slow[i]
        
        # Check for crossover (current vs previous)
        ema_cross_long = ema_bullish and (ema_fast[i-1] <= ema_slow[i-1])
        ema_cross_short = ema_bearish and (ema_fast[i-1] >= ema_slow[i-1])
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > 1.3 * vol_sma[i]
        
        # === RSI FILTER ===
        rsi_long_ok = rsi[i] > 45  # Not oversold, has momentum
        rsi_short_ok = rsi[i] < 55  # Not overbought, has momentum
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG: 4h bullish + EMA cross long + volume + RSI ok
        if bull_trend_4h and ema_cross_long and vol_confirmed and rsi_long_ok:
            new_signal = SIZE
        
        # SHORT: 4h bearish + EMA cross short + volume + RSI ok
        elif bear_trend_4h and ema_cross_short and vol_confirmed and rsi_short_ok:
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
        # Exit if 4h trend reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === EMA REVERSAL EXIT ===
        # Exit if EMA crossover reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and ema_bearish:
                new_signal = 0.0
            if position_side < 0 and ema_bullish:
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