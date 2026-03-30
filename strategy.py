Looking at the experiment history, I see the critical pattern: strategies with >300 trades fail due to fee drag, and those with <50 trades are statistically unreliable. The winning formula needs TIGHT entry conditions + regime filtering.

My hypothesis: A 12h Donchian(40) breakout (10+ day channel) with a stricter 1d EMA21 trend filter + volume confirmation should generate 50-120 trades over 4 years. The longer Donchian period reduces false breakouts, and the EMA21 filter ensures we only trade with the higher timeframe trend.

Key improvements over the current approach:
1. Increase Donchian period from 20 to 40 (reduces trade frequency by ~50%)
2. Use EMA21 instead of SMA50 (faster, more responsive)
3. Add 1d HTF ATR for stoploss scaling
4. Require stronger volume confirmation (1.8x)
5. Use 1d Donchian for structure confirmation (parallel HTF analysis)
#!/usr/bin/env python3
"""
Experiment #028: 12h Donchian(40) Breakout + 1d EMA21 Trend + Volume Spike + ATR Stoploss

HYPOTHESIS: Longer Donchian period (40 = 20-day channel) captures major institutional
breakouts rather than noise. 1d EMA21 provides trend direction. Volume spike confirms
institutional participation. Choppiness filter keeps us out of range-bound markets.

WHY 12h: Slower than 6h/4h, so fewer but higher-quality signals. 50-120 trades over
4 years = manageable fee drag. Works in both bull (long breakouts) and bear (short breakdowns).

TARGET: 50-120 total trades over 4 years. HARD MAX: 200.
Signal size: 0.25 (conservative).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian40_ema21_vol_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - lower = trending, higher = choppy"""
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]))
            else:
                tr = high[j] - low[j]
            tr_sum += tr
        
        if tr_sum > 0:
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            range_hl = hh - ll
            
            if range_hl > 0:
                chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA21 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1d ATR for stoploss scaling
    htf_high = df_1d['high'].values
    htf_low = df_1d['low'].values
    htf_close = df_1d['close'].values
    htf_atr = calculate_atr(htf_high, htf_low, htf_close, period=14)
    htf_atr_aligned = align_htf_to_ltf(prices, df_1d, htf_atr)
    
    # Local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Donchian channels (40 periods = 20 days on 12h)
    donchian_period = 40
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    entry_idx = 0
    
    warmup = 150  # Need enough for Donchian(40) + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA21) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        bull_trend = price_above_1d_ema
        bear_trend = not price_above_1d_ema
        
        # === REGIME (Choppiness Index) ===
        # Skip if too choppy when not in position
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 50.0
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        current_high = high[i]
        current_low = low[i]
        
        # Previous bar's Donchian values
        prev_donchian_high = donchian_high[i - 1] if i > 0 else 0
        prev_donchian_low = donchian_low[i - 1] if i > 0 else 0
        
        # Volume confirmation (stronger = 1.8x)
        vol_spike = vol_ratio[i] > 1.8
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above Donchian high + bull trend + volume ===
            # Price closes above previous 40-bar high with volume spike in bull trend
            if current_high > prev_donchian_high:
                if bull_trend and vol_spike:
                    desired_signal = SIZE
                # Also allow if strongly trending (CHOP < 45)
                elif is_trending and vol_spike:
                    desired_signal = SIZE
            
            # === SHORT: Breakdown below Donchian low + bear trend + volume ===
            if current_low < prev_donchian_low:
                if bear_trend and vol_spike:
                    desired_signal = -SIZE
                elif is_trending and vol_spike:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.0 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT at 2R ===
        if in_position and not stoploss_triggered:
            if position_side > 0:
                profit_target = entry_price + 2.0 * entry_atr
                if close[i] >= profit_target:
                    # Take partial profit, trail remaining
                    desired_signal = SIZE / 2  # Half position
                    # Update stop to breakeven
                    stop_price = max(stop_price, entry_price)
            elif position_side < 0:
                profit_target = entry_price - 2.0 * entry_atr
                if close[i] <= profit_target:
                    # Take partial profit, trail remaining
                    desired_signal = -SIZE / 2  # Half position
                    # Update stop to breakeven
                    stop_price = min(stop_price, entry_price)
        
        # === TIME-BASED EXIT ===
        # Hold at least 8 bars (4 days on 12h) before mid-channel exit
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 8:
            # Exit if price reverts to middle of channel
            if position_side > 0 and close[i] < donchian_mid[i]:
                desired_signal = 0.0
            if position_side < 0 and close[i] > donchian_mid[i]:
                desired_signal = 0.0
        
        # === MAX HOLD (prevent endless holds) ===
        if in_position and bars_held >= 30:  # ~15 days max
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals