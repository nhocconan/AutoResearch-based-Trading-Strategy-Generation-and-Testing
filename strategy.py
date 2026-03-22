#!/usr/bin/env python3
"""
Experiment #550: 1h Primary + 4h/12h HTF — Simplified Triple-Timeframe Pullback

Hypothesis: After 480+ failed strategies, the clearest pattern for lower TF success is:
- 1h strategies failed (#540, #545, #548) because TOO MANY conflicting filters = 0 trades
- Key insight: Use HTF (12h/4h) for DIRECTION, 1h only for ENTRY TIMING
- This gives HTF trade frequency (30-60/year) with 1h execution precision
- Simplified confluence: 12h trend + 4h confirmation + 1h RSI pullback + volume filter
- Session filter (8-20 UTC) avoids Asian session whipsaws (proven in FX/crypto literature)
- Asymmetric sizing: 0.25 bull, 0.20 bear (crypto crashes faster than rallies)

Why this might beat Sharpe=0.435:
- 12h HMA(21) for major trend (slow, reliable direction filter)
- 4h HMA(21) aligned for intermediate confirmation (prevents counter-trend entries)
- 1h RSI(14) pullback: 38-52 for longs, 48-62 for shorts (not extreme = more trades)
- Volume filter: current > 0.8x 20-bar avg (avoids low-liquidity traps)
- Session filter: 8-20 UTC only (London/NY overlap = best liquidity)
- 2.5x ATR trailing stop on all positions
- Target: 40-60 trades/year on 1h (optimal per Rule 10)

Position sizing: 0.20-0.25 base (discrete, max 0.40 per Rule 4)
Stoploss: 2.5 * ATR(14) trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h12h_session_v1"
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

def calculate_sma(close, period=20):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def get_hour_from_open_time(prices):
    """Extract UTC hour from open_time column."""
    # open_time is in milliseconds since epoch
    timestamps = prices['open_time'].values / 1000.0
    hours = (timestamps % 86400) / 3600.0
    return hours.astype(int)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HTF HMA for major trend direction
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_50 = calculate_hma(df_12h['close'].values, period=50)
    
    # Calculate 4h HTF HMA for intermediate confirmation
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_50_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_50)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_sma_20 = calculate_sma(volume, 20)
    
    # Extract UTC hour for session filter
    utc_hours = get_hour_from_open_time(prices)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Asymmetric: smaller size in bear regime (crypto crashes faster)
    POSITION_SIZE_BULL = 0.25
    POSITION_SIZE_BEAR = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_50_aligned[i]):
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_50_aligned[i]):
            continue
        if np.isnan(rsi_14[i]):
            continue
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] == 0:
            continue
        
        # === 12H MAJOR TREND (primary direction filter) ===
        bull_regime_12h = close[i] > hma_12h_21_aligned[i]
        bear_regime_12h = close[i] < hma_12h_21_aligned[i]
        
        # 12h HMA slope for trend strength
        hma_12h_slope_bull = hma_12h_21_aligned[i] > hma_12h_50_aligned[i]
        hma_12h_slope_bear = hma_12h_21_aligned[i] < hma_12h_50_aligned[i]
        
        # === 4H INTERMEDIATE CONFIRMATION ===
        bull_regime_4h = close[i] > hma_4h_21_aligned[i]
        bear_regime_4h = close[i] < hma_4h_21_aligned[i]
        
        hma_4h_slope_bull = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_slope_bear = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # === VOLUME FILTER (avoid low-liquidity entries) ===
        # Current volume must be at least 80% of 20-bar average
        vol_ok = volume[i] > 0.8 * vol_sma_20[i]
        
        # === SESSION FILTER (8-20 UTC only) ===
        # London/NY overlap has best liquidity, avoids Asian whipsaws
        session_ok = 8 <= utc_hours[i] <= 20
        
        # === RSI PULLBACK ENTRY (1h timing) ===
        # Long: RSI 38-52 in uptrend (pullback, not oversold crash)
        rsi_pullback_long = 38.0 <= rsi_14[i] <= 52.0
        # Short: RSI 48-62 in downtrend (rally into resistance)
        rsi_pullback_short = 48.0 <= rsi_14[i] <= 62.0
        
        # === ENTRY LOGIC — TRIPLE TIMEFRAME CONFLUENCE ===
        new_signal = 0.0
        
        # LONG ENTRY: 12h bull + 4h bull + RSI pullback + vol OK + session OK
        if (bull_regime_12h and bull_regime_4h and rsi_pullback_long and 
            vol_ok and session_ok):
            # Size based on 12h regime strength
            if hma_12h_slope_bull:
                new_signal = POSITION_SIZE_BULL
            else:
                new_signal = POSITION_SIZE_BULL * 0.8
        
        # SHORT ENTRY: 12h bear + 4h bear + RSI pullback + vol OK + session OK
        elif (bear_regime_12h and bear_regime_4h and rsi_pullback_short and 
              vol_ok and session_ok):
            # Size based on 12h regime strength
            if hma_12h_slope_bear:
                new_signal = -POSITION_SIZE_BEAR
            else:
                new_signal = -POSITION_SIZE_BEAR * 0.8
        
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
        
        # === EXIT CONDITIONS (regime flip) ===
        # Exit long on 12h regime flip to bear
        if in_position and position_side > 0:
            if bear_regime_12h and hma_12h_slope_bear:
                new_signal = 0.0
            elif bear_regime_4h and hma_4h_slope_bear:
                new_signal = 0.0
        
        # Exit short on 12h regime flip to bull
        if in_position and position_side < 0:
            if bull_regime_12h and hma_12h_slope_bull:
                new_signal = 0.0
            elif bull_regime_4h and hma_4h_slope_bull:
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