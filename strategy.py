#!/usr/bin/env python3
"""
Experiment #024: 4h Donchian Breakout + Choppiness Regime + Williams Alligator + Volume

HYPOTHESIS: Price channel breakouts (Donchian) identify trend shifts reliably.
Combined with:
- Williams Alligator for smooth trend direction (no lag vs EMA)
- Choppiness Index to avoid whipsaws in ranging markets
- Volume confirmation to filter false breakouts

WHY IT WORKS BOTH MARKETS:
- 2021 bull: Breakout above Alligator in chop < 38.2 = strong trend
- 2022 bear: Breakout below Alligator in chop < 38.2 = strong short
- 2025 range: Choppiness > 61.8 = no trades (avoids whipsaws)
- ATR stoploss protects against 2022 crash

TIMEFRAME: 4h (proven in DB - best performers use 4h)
TARGET TRADES: 75-200 total over 4 years (19-50/year)
Size: 0.30
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_chop_alligator_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_alligator(high, low, period=13):
    """
    Williams Alligator: 3 smoothed moving averages
    Jaw (blue): SMA(13) of median, shifted 8 bars forward
    Teeth (red): SMA(8) of median, shifted 5 bars forward
    Lips (green): SMA(5) of median, shifted 3 bars forward
    
    Trend = bullish when Jaw > Teeth > Lips
    Trend = bearish when Jaw < Teeth < Lips
    """
    median = (high + low) / 2.0
    
    jaw = pd.Series(median).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median).rolling(window=5, min_periods=5).mean().shift(3).values
    
    return jaw, teeth, lips

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (Chop):
    - Values > 61.8 = ranging market (mean reversion works better)
    - Values < 38.2 = trending market (trend following works better)
    
    We want chop < 38.2 for trend entries.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest - lowest > 1e-10:
            sum_tr = 0.0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
                sum_tr += tr
            
            chop[i] = 100 * np.log10(sum_tr / (highest - lowest)) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian channel - price channel breakout"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d EMA for macro trend (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d_21 = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_21)
    
    # === 4h indicators (pre-compute before loop) ===
    jaw, teeth, lips = calculate_alligator(high, low)
    chop = calculate_choppiness(high, low, close, period=14)
    dc_upper_20, dc_lower_20 = calculate_donchian(high, low, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume analysis
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 60  # Need at least 60 bars for Alligator + ATR warmup
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === TREND DETECTION ===
        # 1d EMA for macro direction
        htf_bullish = close[i] > ema_1d_aligned[i]
        htf_bearish = close[i] < ema_1d_aligned[i]
        
        # Williams Alligator trend
        jaw_val = jaw[i] if not np.isnan(jaw[i]) else 0
        teeth_val = teeth[i] if not np.isnan(teeth[i]) else 0
        lips_val = lips[i] if not np.isnan(lips[i]) else 0
        
        # Bullish: lips > teeth > jaw
        alligator_bullish = lips_val > teeth_val > jaw_val
        # Bearish: lips < teeth < jaw
        alligator_bearish = lips_val < teeth_val < jaw_val
        
        # Choppiness regime
        is_trending = chop[i] < 38.2  # Trending market
        is_ranging = chop[i] > 61.8    # Ranging market - skip
        
        # === DONCHIAN BREAKOUT ===
        dc_upper = dc_upper_20[i] if not np.isnan(dc_upper_20[i]) else 0
        dc_lower = dc_lower_20[i] if not np.isnan(dc_lower_20[i]) else 0
        
        # Breakout above upper band
        breakout_up = close[i] > dc_upper and close[i-1] <= dc_upper
        # Breakdown below lower band
        breakout_down = close[i] < dc_lower and close[i-1] >= dc_lower
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === MINIMUM HOLD: 2 bars (8h) ===
        min_hold = (i - entry_bar) >= 2
        
        # === ATR TRAILING STOP (2.5x ATR) ===
        def check_atr_stop():
            if not in_position:
                return False
            if position_side > 0:
                return low[i] < (highest_since_entry - 2.5 * entry_atr)
            else:
                return high[i] > (lowest_since_entry + 2.5 * entry_atr)
        
        # === EXITS ===
        if in_position:
            stop_hit = check_atr_stop()
            
            # Trend reversal exits (Alligator flip)
            if position_side > 0 and alligator_bearish and min_hold:
                stop_hit = True
            if position_side < 0 and alligator_bullish and min_hold:
                stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # Skip if ranging (choppiness > 61.8)
            if is_ranging:
                signals[i] = 0.0
                continue
            
            # LONG: Breakout above Donchian + Alligator bullish + Trending + volume
            if breakout_up and alligator_bullish and is_trending and vol_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # LONG WEAKER: Alligator bullish + Trending + volume + close above all 3 lines
            elif alligator_bullish and is_trending and vol_spike and close[i] > lips_val:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE * 0.5  # Half size - not breakout confirmed
            
            # SHORT: Breakdown below Donchian + Alligator bearish + Trending + volume
            elif breakout_down and alligator_bearish and is_trending and vol_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            # SHORT WEAKER: Alligator bearish + Trending + volume + close below all 3 lines
            elif alligator_bearish and is_trending and vol_spike and close[i] < lips_val:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE * 0.5  # Half size - not breakdown confirmed
            
            else:
                signals[i] = 0.0
    
    return signals