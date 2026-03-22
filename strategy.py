#!/usr/bin/env python3
"""
Experiment #429: 1h RSI Pullback + 4h HMA Trend + Volume Confirmation

Hypothesis: After 428 failed experiments, the pattern is clear - complex regime
switching strategies (ADX+Chop+Bollinger) are overfitting and failing. The key
insight is SIMPLICITY + STRONG HTF BIAS. This strategy uses:

1. 4h HMA(21) TREND BIAS (via mtf_data helper):
   - Long ONLY when price > 4h HMA (bullish bias)
   - Short ONLY when price < 4h HMA (bearish bias)
   - HMA is smoother than EMA, reduces whipsaw on 1h entries
   - This is the SINGLE strongest filter - no counter-trend trades

2. RSI(7) PULLBACK ENTRY (faster than RSI(14) for 1h):
   - Long: RSI(7) < 35 (oversold pullback in uptrend)
   - Short: RSI(7) > 65 (overbought pullback in downtrend)
   - Tighter thresholds than standard 30/70 for more trades

3. VOLUME CONFIRMATION (critical filter):
   - Entry volume > 1.5x 20-bar average volume
   - Confirms institutional interest, reduces false signals
   - This was underutilized in failed strategies

4. EMA(21) PROXIMITY FILTER:
   - Long: price within 2% of EMA(21) (pullback, not breakout)
   - Short: price within 2% of EMA(21)
   - Ensures we're buying dips, not chasing

5. ATR(14) TRAILING STOP at 2.0x:
   - Signal → 0 when price moves 2.0*ATR against position
   - Tighter than 2.5x for 1h timeframe (faster exits)

6. POSITION SIZING: 0.25 discrete (conservative for 1h volatility)
   - Max 25% capital per position
   - Discrete levels: 0.0, ±0.25 only (minimize fee churn)

Why 1h should work:
- More trades than 4h/12h (~50-100/year vs 20-40)
- Faster reaction to reversals than 4h strategies
- 4h HMA provides stable trend bias without overfitting
- Volume filter is unique - most failed strategies ignored volume
- RSI(7) is fast enough for 1h but not noisy like RSI(3)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi7_pullback_4h_hma_vol_confirm_atr_v1"
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

def calculate_rsi(close, period=7):
    """Calculate Relative Strength Index with custom period."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_volume_sma(volume, period=20):
    """Calculate simple moving average of volume."""
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
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 7)
    ema_21 = calculate_ema(close, 21)
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
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === 4h HMA TREND BIAS (SINGLE STRONGEST FILTER) ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = volume[i] > 1.5 * vol_sma[i]
        
        # === EMA PROXIMITY FILTER (pullback, not breakout) ===
        ema_pct_diff = abs(close[i] - ema_21[i]) / ema_21[i]
        near_ema = ema_pct_diff < 0.02  # Within 2% of EMA
        
        # === RSI PULLBACK SIGNALS ===
        rsi_oversold = rsi[i] < 35  # Long entry
        rsi_overbought = rsi[i] > 65  # Short entry
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG: 4h bullish + RSI oversold + volume spike + near EMA
        if bull_trend_4h and rsi_oversold and vol_spike and near_ema:
            new_signal = SIZE
        
        # SHORT: 4h bearish + RSI overbought + volume spike + near EMA
        elif bear_trend_4h and rsi_overbought and vol_spike and near_ema:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 4h trend turns bearish
        if in_position and position_side > 0 and bear_trend_4h:
            new_signal = 0.0
        
        # Exit short if 4h trend turns bullish
        if in_position and position_side < 0 and bull_trend_4h:
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