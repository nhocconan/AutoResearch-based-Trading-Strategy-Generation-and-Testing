#!/usr/bin/env python3
"""
Experiment #540: 1h Primary + 4h/12h HTF — HMA Trend + RSI Pullback + Volume + Session

Hypothesis: After 480+ failed strategies, 1h timeframe can work with EXTREMELY strict
entry filters to limit trades to 30-80/year. Key is using HTF (4h/12h) for SIGNAL
DIRECTION and 1h only for ENTRY TIMING.

Key insights from failures:
- #530, #535, #538: 1h/30m strategies failed with Sharpe=0.000 or negative
- #536: 12h strategy worked (Sharpe=0.159) — proves MTF trend+pullback works
- Lower TF needs 3+ confluence filters to avoid fee drag (>100 trades/year = death)

This strategy uses:
1. 12h HMA(21) for MAJOR trend direction (strict HTF filter)
2. 4h HMA(21) for INTERMEDIATE trend confirmation
3. 1h RSI(14) pullback entries (30-45 for long, 55-70 for short)
4. Volume filter: current > 0.8x 20-bar avg (avoid low-liquidity traps)
5. Session filter: only 8-20 UTC (high-liquidity hours)
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete position sizing (0.25) to minimize fee churn

Why this might work on 1h:
- 12h HMA prevents counter-trend trades (major failure mode on lower TF)
- 4h HMA adds intermediate confirmation (reduces false signals)
- RSI pullback avoids chasing breakouts (proven in #536)
- Volume + Session filters cut low-quality trades by ~60%
- 1h entries within HTF trend = HTF frequency with 1h precision
- Simple logic = consistent signals across BTC/ETH/SOL

Position sizing: 0.25 (conservative for 1h, max 0.40 per rules)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 30-80 trades/symbol/year on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_volume_session_4h12h_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = volume / (vol_avg.values + 1e-10)
    return vol_ratio

def extract_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HTF HMA for MAJOR trend direction
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_50 = calculate_hma(df_12h['close'].values, period=50)
    
    # Calculate 4h HTF HMA for INTERMEDIATE trend
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_50_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_50)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Extract session hours
    hours = np.array([extract_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, conservative for 1h)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track HMA slopes
    prev_hma_4h = np.zeros(n)
    prev_hma_4h[1:] = hma_4h_21_aligned[:-1]
    
    prev_hma_12h = np.zeros(n)
    prev_hma_12h[1:] = hma_12h_21_aligned[:-1]
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_50_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(rsi_14[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === 12H MAJOR TREND (primary direction filter — MUST align) ===
        bull_regime_12h = close[i] > hma_12h_21_aligned[i]
        bear_regime_12h = close[i] < hma_12h_21_aligned[i]
        
        # 12h HMA slope for trend strength
        hma_12h_slope_bull = hma_12h_21_aligned[i] > hma_12h_50_aligned[i]
        hma_12h_slope_bear = hma_12h_21_aligned[i] < hma_12h_50_aligned[i]
        
        # === 4H INTERMEDIATE TREND (confirmation filter) ===
        bull_regime_4h = close[i] > hma_4h_21_aligned[i]
        bear_regime_4h = close[i] < hma_4h_21_aligned[i]
        
        hma_4h_slope_bull = hma_4h_21_aligned[i] > prev_hma_4h[i]
        hma_4h_slope_bear = hma_4h_21_aligned[i] < prev_hma_4h[i]
        
        # === SESSION FILTER (8-20 UTC only — high liquidity) ===
        in_session = (hours[i] >= 8) and (hours[i] <= 20)
        
        # === VOLUME FILTER (avoid low-liquidity traps) ===
        volume_ok = vol_ratio[i] > 0.8
        
        # === RSI PULLBACK FILTER (strict ranges for few trades) ===
        rsi_pullback_long = 30.0 <= rsi_14[i] <= 45.0
        rsi_pullback_short = 55.0 <= rsi_14[i] <= 70.0
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === ENTRY LOGIC — 4+ CONFLUENCE FOR LOW TRADE COUNT ===
        new_signal = 0.0
        
        # LONG ENTRIES (ALL conditions must align — strict for 1h)
        # Condition 1: 12h bull + 4h bull + RSI pullback + session + volume
        if (bull_regime_12h and hma_12h_slope_bull and 
            bull_regime_4h and hma_4h_slope_bull and
            rsi_pullback_long and in_session and volume_ok):
            new_signal = POSITION_SIZE
        
        # Condition 2: 12h bull + 4h bull + RSI oversold (deep pullback)
        elif (bull_regime_12h and hma_12h_slope_bull and
              bull_regime_4h and rsi_oversold and in_session):
            new_signal = POSITION_SIZE
        
        # Condition 3: Strong 12h trend + 4h pullback to HMA + RSI recovering
        elif (bull_regime_12h and hma_12h_slope_bull and
              close[i] > hma_4h_21_aligned[i] * 0.98 and  # near 4h HMA
              35.0 <= rsi_14[i] <= 50.0 and in_session and volume_ok):
            new_signal = POSITION_SIZE * 0.8
        
        # SHORT ENTRIES (mirror logic)
        if new_signal == 0.0:
            # Condition 1: 12h bear + 4h bear + RSI pullback + session + volume
            if (bear_regime_12h and hma_12h_slope_bear and
                bear_regime_4h and hma_4h_slope_bear and
                rsi_pullback_short and in_session and volume_ok):
                new_signal = -POSITION_SIZE
            
            # Condition 2: 12h bear + 4h bear + RSI overbought (deep bounce)
            elif (bear_regime_12h and hma_12h_slope_bear and
                  bear_regime_4h and rsi_overbought and in_session):
                new_signal = -POSITION_SIZE
            
            # Condition 3: Strong 12h trend + 4h bounce to HMA + RSI rolling over
            elif (bear_regime_12h and hma_12h_slope_bear and
                  close[i] < hma_4h_21_aligned[i] * 1.02 and  # near 4h HMA
                  50.0 <= rsi_14[i] <= 65.0 and in_session and volume_ok):
                new_signal = -POSITION_SIZE * 0.8
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS (regime flip or trend weakening) ===
        # Exit long on 12h regime flip to bear
        if in_position and position_side > 0:
            if bear_regime_12h and hma_12h_slope_bear:
                new_signal = 0.0
        
        # Exit short on 12h regime flip to bull
        if in_position and position_side < 0:
            if bull_regime_12h and hma_12h_slope_bull:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals