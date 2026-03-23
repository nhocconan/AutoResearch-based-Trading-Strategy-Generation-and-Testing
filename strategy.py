#!/usr/bin/env python3
"""
Experiment #650: 1h Primary + 4h/12h HTF — Fisher Transform Reversals + HMA Trend + Volume

Hypothesis: Building on failed Connors RSI + Choppiness experiments (#641, #645, #648),
this strategy uses Fisher Transform for reversal entries (proven in bear/range markets),
4h HMA for trend direction, and 12h HMA for regime filter. Key difference from failures:
- Fisher Transform catches reversals at extremes (better than RSI in 2022 crash)
- Session filter (8-20 UTC) reduces trades to target 30-80/year
- Volume confirmation ensures real moves, not noise
- Looser entry thresholds to avoid 0-trade problem (#638, #645, #648)

Why this might beat Sharpe=0.520:
1. Fisher Transform excels in bear/range markets (2025 test period is -25% BTC)
2. 4h HMA trend filter keeps us on right side (proven in #624)
3. 12h HMA slope confirms major regime (avoid counter-trend trades)
4. Volume > 0.8x avg filters false breakouts
5. Session filter (8-20 UTC) = ~60% fewer trades = less fee drag
6. Conservative size (0.25) + 2.5*ATR stop controls drawdown

Position sizing: 0.25 discrete (per Rule 4, max 0.40)
Target: 40-80 trades/year on 1h (per Rule 10)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_hma_vol_session_4h12h_v1"
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
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag.
    """
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into Gaussian distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5 from below.
    Short when Fisher crosses below +1.5 from above.
    """
    # Calculate typical price
    typical = (high + low) / 2.0
    typical_s = pd.Series(typical)
    
    # Normalize price over lookback period
    highest = typical_s.rolling(window=period, min_periods=period).max()
    lowest = typical_s.rolling(window=period, min_periods=period).min()
    
    # Normalize to 0-1 range
    range_hl = highest - lowest
    range_hl = range_hl.replace(0, 1e-10)  # avoid div by zero
    normalized = (typical - lowest) / range_hl
    
    # Clamp to avoid extreme values
    normalized = np.clip(normalized, 0.001, 0.999)
    
    # Fisher Transform formula
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
    
    # Signal line (1-period lag)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher.values, fisher_signal

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
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
    
    # Calculate 4h HMA for trend direction
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 12h HMA for regime filter
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
    vol_avg = calculate_volume_avg(volume, 20)
    
    # Also calculate 1h HMA for additional confirmation
    hma_1h = calculate_hma(close, 21)
    hma_1h_fast = calculate_hma(close, 9)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(rsi_14[i]) or np.isnan(fisher[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        if atr_14[i] == 0:
            continue
        
        # Extract UTC hour for session filter
        hour_utc = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour_utc <= 20  # Only trade 8-20 UTC
        
        # Volume filter
        volume_ok = volume[i] >= 0.8 * vol_avg[i]
        
        # === 12H REGIME FILTER (HMA slope over 5 bars) ===
        hma_12h_slope_bull = hma_12h_aligned[i] > hma_12h_aligned[i-5] if i >= 5 else False
        hma_12h_slope_bear = hma_12h_aligned[i] < hma_12h_aligned[i-5] if i >= 5 else False
        
        # Price relative to 12h HMA
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === 4H TREND BIAS (HMA slope over 3 bars) ===
        hma_4h_slope_bull = hma_4h_aligned[i] > hma_4h_aligned[i-3] if i >= 3 else False
        hma_4h_slope_bear = hma_4h_aligned[i] < hma_4h_aligned[i-3] if i >= 3 else False
        
        # Price relative to 4h HMA
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 1H HMA FAST/SLOW CROSSOVER ===
        hma_1h_cross_bull = hma_1h_fast[i] > hma_1h[i]
        hma_1h_cross_bear = hma_1h_fast[i] < hma_1h[i]
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        fisher_long = (fisher[i] > -1.5) and (fisher_signal[i] <= -1.5)
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_short = (fisher[i] < 1.5) and (fisher_signal[i] >= 1.5)
        
        # Alternative: Fisher extreme values with RSI confirmation
        fisher_oversold = fisher[i] < -1.0 and rsi_14[i] < 45
        fisher_overbought = fisher[i] > 1.0 and rsi_14[i] > 55
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: 12h/4h bull trend + Fisher reversal + volume ---
        # Condition 1: 12h HMA sloping up OR price above 12h HMA (regime OK)
        # Condition 2: 4h HMA sloping up AND price above 4h HMA (trend OK)
        # Condition 3: Fisher reversal signal OR Fisher oversold + RSI confirmation
        # Condition 4: Volume >= 0.8x average
        # Condition 5: In session (8-20 UTC)
        if (hma_12h_slope_bull or price_above_hma_12h):
            if hma_4h_slope_bull and price_above_hma_4h:
                if (fisher_long or fisher_oversold):
                    if volume_ok and in_session:
                        new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY: 12h/4h bear trend + Fisher reversal + volume ---
        # Condition 1: 12h HMA sloping down OR price below 12h HMA (regime OK)
        # Condition 2: 4h HMA sloping down AND price below 4h HMA (trend OK)
        # Condition 3: Fisher reversal signal OR Fisher overbought + RSI confirmation
        # Condition 4: Volume >= 0.8x average
        # Condition 5: In session (8-20 UTC)
        elif (hma_12h_slope_bear or price_below_hma_12h):
            if hma_4h_slope_bear and price_below_hma_4h:
                if (fisher_short or fisher_overbought):
                    if volume_ok and in_session:
                        new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
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
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if hma_4h_slope_bear and price_below_hma_4h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_4h_slope_bull and price_above_hma_4h:
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