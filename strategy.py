Looking at the experiment history, I see the key patterns:
1. **Overtrading** is the #1 killer (1426, 1155, 431 trades all failed)
2. **Too few trades** is #2 (0 trades, 1 trade)
3. **Successful strategies** had 75-300 tight-filtered trades

The current 6h strategy might be too loose with only 1.5x volume + trending mode. Let me try a **tighter ATR channel breakout with stricter filters** to ensure proper trade count.

**New hypothesis**: Use dynamic ATR bands with STRICTER 2.0x volume spike AND 1d ATR ratio filter to ensure we're in a volatility expansion environment (not chop). This should reduce false breakouts and keep trades in the sweet spot.
#!/usr/bin/env python3
"""
Experiment #028: 4h ATR Channel Breakout + Volatility Expansion Filter

HYPOTHESIS: ATR-based channels adapt dynamically to volatility. By combining
channel breakouts with a VOLATILITY EXPANSION filter (current ATR > 1.5x 20d avg ATR)
and volume confirmation, we capture only high-momentum breakouts when volatility
is expanding — the ideal environment for trend continuation.

WHY IT WORKS IN BULL AND BEAR: ATR channels are symmetrical. Shorting breakdowns
below lower band works in bear. Long breakouts above upper band works in bull.
Volatility expansion filter keeps us out of low-volatility range traps.
1d EMA50 ensures higher-timeframe trend alignment.

TARGET: 60-150 total trades over 4 years (15-37/year).
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_atr_channel_vol_expansion_1d_v1"
timeframe = "4h"
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
    
    # 1d EMA50 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1d ATR for volatility expansion filter
    atr_1d = calculate_atr(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values, 
        period=14
    )
    atr_1d_ma = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_1d_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_ma)
    
    # Local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_14_ma = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # ATR channels (20 periods = ~5 day channel on 4h)
    channel_period = 20
    channel_mult = 2.0
    channel_upper = pd.Series(close).rolling(window=channel_period, min_periods=channel_period).mean().values + channel_mult * atr_14
    channel_lower = pd.Series(close).rolling(window=channel_period, min_periods=channel_period).mean().values - channel_mult * atr_14
    
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
    
    warmup = 150  # Need enough for all indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or np.isnan(atr_1d_ma_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === VOLATILITY EXPANSION FILTER ===
        # Only trade when ATR is expanding (volatility picking up)
        vol_expanding = atr_1d_aligned[i] > 1.3 * atr_1d_ma_aligned[i]
        
        # === CHOP FILTER ===
        # Skip if too choppy
        is_choppy = chop[i] > 61.8
        
        if is_choppy and not in_position:
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION (1d EMA50) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # === CHANNEL VALUES ===
        current_upper = channel_upper[i]
        current_lower = channel_lower[i]
        prev_upper = channel_upper[i - 1] if i > 0 else current_upper
        prev_lower = channel_lower[i - 1] if i > 0 else current_lower
        
        # === VOLUME CONFIRMATION (STRICT - 2.0x) ===
        vol_spike = vol_ratio[i] > 2.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above upper ATR channel ===
            # Price closes above channel with volume + volatility expansion
            if close[i] > prev_upper and price_above_1d_ema:
                if vol_spike and vol_expanding:
                    desired_signal = SIZE
            
            # === SHORT: Breakdown below lower ATR channel ===
            if close[i] < prev_lower and not price_above_1d_ema:
                if vol_spike and vol_expanding:
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
        
        # === TIME-BASED EXIT (hold at least 4 bars = ~16h) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 4:
            # Exit if price reverts to channel middle
            channel_mid = (current_upper + current_lower) / 2
            if position_side > 0 and close[i] < channel_mid:
                desired_signal = 0.0
            if position_side < 0 and close[i] > channel_mid:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
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