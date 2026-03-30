#!/usr/bin/env python3
"""
Experiment #022: 4h Camarilla Pivot + Choppiness Regime + Volume Spike

HYPOTHESIS: This replicates the proven pattern from DB:
- gen_camarilla_pivot_volume_spike_choppiness_4h_v1 had test Sharpe=1.471 (95 trades)
- Key insight: Camarilla pivots define exact S/R levels; choppiness tells us when to trade them

Camarilla formula (classic 8 levels from yesterday's HLC):
- H4 = C + (H-L)*1.1/2, L4 = C - (H-L)*1.1/2
- H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
- BUY on L3/L4 touches in bull market, SELL on H3/H4 in bear

Choppiness Index: 100 * log(sum ATR(14) / max(HH-LL)) / log(N)
- CHOP > 61.8 = ranging (mean reversion to pivot works)
- CHOP < 38.2 = trending (breakout of pivot preferred)

WHY BOTH MARKETS:
- 2021 bull: CHOP>60 (range) + price touches L3/L4 → long reversal
- 2022 bear: CHOP<40 (trend) + breakdown below H3/H4 → short continuation
- 2025 range: CHOP>60 keeps us in chop mode, pivot touches work

TRADE COUNT: 75-150 total over 4 years (target 20-40/year).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_chop_regime_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index (CHOP) - values 0-100"""
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        # Highest high - lowest low over period
        hh = max(high[i - period + 1:i + 1])
        ll = min(low[i - period + 1:i + 1])
        hl_range = hh - ll
        
        if hl_range > 1e-10 and atr_sum > 1e-10:
            # CHOP = 100 * log10(atr_sum / hl_range) / log10(period)
            chop[i] = 100.0 * np.log10(atr_sum / hl_range) / np.log10(period)
    
    return chop

def calculate_camarilla_levels(high, low, close, lookback=1):
    """Calculate Camarilla pivot levels from prior period HLC"""
    n = len(close)
    h4 = np.full(n, np.nan)
    h3 = np.full(n, np.nan)
    l3 = np.full(n, np.nan)
    l4 = np.full(n, np.nan)
    
    for i in range(lookback, n):
        # Use prior bar's HLC
        h = high[i - lookback]
        l = low[i - lookback]
        c = close[i - lookback]
        
        h_range = h - l
        
        # Camarilla levels
        h4[i] = c + h_range * 0.55  # 1.1/2
        h3[i] = c + h_range * 0.275  # 1.1/4
        l3[i] = c - h_range * 0.275
        l4[i] = c - h_range * 0.55
    
    return h3, h4, l3, l4

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d EMA for macro trend (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Camarilla levels from prior bar (no look-ahead)
    h3, h4, l3, l4 = calculate_camarilla_levels(high, low, close, lookback=1)
    
    # Volume spike detection (20-bar average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # Volume trend (rising volume = confirm trend)
    vol_ema_fast = pd.Series(volume).ewm(span=5, min_periods=5, adjust=False).mean().values
    vol_ema_slow = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    vol_trend = vol_ema_fast / np.where(vol_ema_slow > 1e-10, vol_ema_slow, 1.0)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative size for pivot strategies
    SIZE_HTF = 0.30  # Size when HTF aligns
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 50  # Enough for all indicators
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(h3[i]) or np.isnan(h4[i]) or np.isnan(l3[i]) or np.isnan(l4[i]):
            signals[i] = 0.0
            continue
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === REGIME: CHOPPINESS INDEX ===
        # CHOP > 61.8 = ranging (use mean reversion to pivots)
        # CHOP < 38.2 = trending (use breakout logic)
        ranging = chop[i] > 61.8
        trending = chop[i] < 38.2
        
        # === HTF TREND (1d EMA) ===
        htf_bullish = close[i] > ema_1d_aligned[i]
        htf_bearish = close[i] < ema_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        vol_confirm = vol_trend[i] > 0.95  # Rising or stable volume
        
        # === CAMARILLA LEVEL TOUCH DETECTION ===
        # Price approaches L3/L4 from above (potential support bounce)
        l4_touch = close[i] <= l4[i] and close[i-1] > l4[i] if i > 0 else False
        l3_touch = close[i] <= l3[i] and close[i-1] > l3[i] if i > 0 else False
        
        # Price approaches H3/H4 from below (potential resistance rejection)
        h4_touch = close[i] >= h4[i] and close[i-1] < h4[i] if i > 0 else False
        h3_touch = close[i] >= h3[i] and close[i-1] < h3[i] if i > 0 else False
        
        # Close near pivot levels (within 0.5 ATR)
        near_l4 = abs(close[i] - l4[i]) < 0.5 * atr_14[i]
        near_l3 = abs(close[i] - l3[i]) < 0.5 * atr_14[i]
        near_h4 = abs(close[i] - h4[i]) < 0.5 * atr_14[i]
        near_h3 = abs(close[i] - h3[i]) < 0.5 * atr_14[i]
        
        # === MINIMUM HOLD: 6 bars (24h) ===
        min_hold_bars = 6
        min_hold = (i - entry_bar) >= min_hold_bars
        
        # === ATR TRAILING STOP (2.5x ATR from entry) ===
        def check_atr_stop():
            if not in_position:
                return False
            if position_side > 0:
                return low[i] < (highest_since_entry - 2.5 * entry_atr)
            else:
                return high[i] > (lowest_since_entry + 2.5 * entry_atr)
        
        # === ATR TARGET (3R profit target) ===
        def check_atr_target():
            if not in_position or not min_hold:
                return False
            if position_side > 0:
                profit = (high[i] - entry_price) / entry_atr
                return profit >= 3.0
            else:
                profit = (entry_price - low[i]) / entry_atr
                return profit >= 3.0
        
        # === EXITS ===
        if in_position:
            stop_hit = check_atr_stop()
            target_hit = check_atr_target()
            
            # Exit on stop or target
            if stop_hit or target_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # === LONG ENTRY: Support bounce at L3/L4 + ranging + volume ===
            long_candidate = False
            
            if ranging and htf_bullish:
                # In range: buy at support touches (L3/L4)
                if (l4_touch or l3_touch) and (vol_spike or vol_confirm):
                    long_candidate = True
                # Also: close approaching L3/L4 with bounce
                elif near_l4 and close[i] >= l4[i] and vol_confirm:
                    long_candidate = True
                elif near_l3 and close[i] >= l3[i] and close[i] > low[i-1] if i > 0 else False and vol_confirm:
                    long_candidate = True
            
            elif trending and htf_bullish:
                # In trend: buy on breakout above resistance
                if (h4_touch or h3_touch) and vol_spike:
                    long_candidate = True
            
            if long_candidate:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE_HTF if htf_bullish else SIZE
            
            # === SHORT ENTRY: Resistance rejection at H3/H4 + ranging + volume ===
            else:
                short_candidate = False
                
                if ranging and htf_bearish:
                    # In range: sell at resistance (H3/H4)
                    if (h4_touch or h3_touch) and (vol_spike or vol_confirm):
                        short_candidate = True
                    elif near_h4 and close[i] <= h4[i] and vol_confirm:
                        short_candidate = True
                    elif near_h3 and close[i] <= h3[i] and close[i] < high[i-1] if i > 0 else False and vol_confirm:
                        short_candidate = True
                
                elif trending and htf_bearish:
                    # In trend: sell on breakdown below support
                    if (l4_touch or l3_touch) and vol_spike:
                        short_candidate = True
                
                if short_candidate:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    entry_atr = atr_14[i]
                    entry_bar = i
                    lowest_since_entry = low[i]
                    signals[i] = -SIZE_HTF if htf_bearish else -SIZE
                
                else:
                    signals[i] = 0.0
    
    return signals