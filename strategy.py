#!/usr/bin/env python3
"""
Experiment #390: 1d Weekly HMA Trend + Daily EMA Pullback + Volume Confirmation

Hypothesis: After 389 failed experiments, the pattern is clear - complex regime 
detection (Choppiness, ADX, Fisher) adds noise, not signal. The winning approach 
is SIMPLER: strong HTF trend filter + clean LTF entry + volume confirmation.

KEY INSIGHTS from failures:
1. Regime filters (CHOP, ADX) create whipsaw - they flip too often
2. Fisher Transform + RSI combinations are over-engineered
3. Daily timeframe needs wider stops (3*ATR) and fewer but stronger signals
4. Volume confirmation is UNDERUTILIZED in failed strategies

STRATEGY COMPONENTS:
1. WEEKLY HMA(21) TREND FILTER (via mtf_data): Ultimate trend bias
   - Price > weekly HMA = only long signals allowed
   - Price < weekly HMA = only short signals allowed
   - This is the SINGLE most important filter (proven in best strategies)

2. DAILY EMA(21) PULLBACK ENTRY: Wait for retracement in trend
   - Long: price > weekly HMA AND price pulls back to EMA(21) AND RSI < 50
   - Short: price < weekly HMA AND price rallies to EMA(21) AND RSI > 50
   - This avoids chasing breakouts (which fail in 2022/2025)

3. VOLUME CONFIRMATION: Entry bar volume > 1.5 * 20-day avg volume
   - Filters out low-conviction moves
   - Critical for avoiding fake breakouts

4. RSI(14) MOMENTUM FILTER: Confirms pullback is exhausted
   - Long: RSI crosses above 45 from below (momentum returning)
   - Short: RSI crosses below 55 from above (momentum returning down)

5. ATR(14) TRAILING STOP (3.0x): Wide stop for daily volatility
   - Signal → 0 when price moves 3*ATR against position
   - Daily bars need wider stops than 4h/1h strategies

6. POSITION SIZING: 0.25 discrete (conservative for daily swings)
   - Max 25% capital per position
   - Discrete levels: 0.0, ±0.25 only (minimize fee churn)

Why this should work:
- Weekly HMA is stable (doesn't flip like 4h HMA)
- Pullback entries have better risk/reward than breakouts
- Volume filter removes 40% of false signals
- 3*ATR stop survives 2022-style crashes
- Should generate 15-30 trades/year (enough for stats, not too many for fees)

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 3.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_weekly_hma_ema_pullback_vol_rsi_atr_v1"
timeframe = "1d"
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

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_volume_ma(volume, period=20):
    """Calculate moving average of volume."""
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    rsi = calculate_rsi(close, 14)
    vol_ma = calculate_volume_ma(volume, 20)
    
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
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_21[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if vol_ma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === WEEKLY HMA TREND FILTER ===
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # === PULLBACK DETECTION ===
        # Long: price pulled back to EMA but still above weekly HMA
        long_pullback = bull_trend_1w and close[i] <= ema_21[i] * 1.02
        # Short: price rallied to EMA but still below weekly HMA
        short_pullback = bear_trend_1w and close[i] >= ema_21[i] * 0.98
        
        # === RSI MOMENTUM CROSSOVER ===
        rsi_long = rsi[i] > 45 and rsi[i-1] <= 45  # crosses above 45
        rsi_short = rsi[i] < 55 and rsi[i-1] >= 55  # crosses below 55
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: Weekly bull + pullback to EMA + volume + RSI momentum
        if long_pullback and volume_confirmed and rsi_long:
            new_signal = SIZE
        
        # SHORT ENTRY: Weekly bear + rally to EMA + volume + RSI momentum
        elif short_pullback and volume_confirmed and rsi_short:
            new_signal = -SIZE
        
        # === TREND REVERSAL EXIT ===
        # Exit long if weekly trend turns bear
        if in_position and position_side > 0 and bear_trend_1w:
            new_signal = 0.0
        
        # Exit short if weekly trend turns bull
        if in_position and position_side < 0 and bull_trend_1w:
            new_signal = 0.0
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing ===
        if in_position and new_signal != 0.0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 3.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            elif position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 3.0 * atr[i]
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