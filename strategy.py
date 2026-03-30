#!/usr/bin/env python3
"""
Experiment #028: 1d ATR Channel Breakout + Williams %R + SMA200 Trend

HYPOTHESIS: Combining volatility-adjusted ATR channels with Williams %R momentum
and 1d SMA200 trend filter creates a swing trading system that captures medium-term
moves while avoiding whipsaws. ATR channels widen during high volatility, reducing
false breakouts. Williams %R confirms momentum exhaustion at channel boundaries.

WHY 1d: Lowest fee drag possible. 10-30 trades/year = ~0.3-1% annual fee cost.
1d SMA200 is the strongest trend filter in crypto (catches 2022 crash, 2024 rally).
1d timeframe has proven stable across bull/bear cycles.

ENTRY CONDITIONS (tight = fewer trades):
- Long: Price breaks above 20d ATR channel high + Williams %R > -20 (momentum)
- Short: Price breaks below 20d ATR channel low + Williams %R < -80 (momentum)
- Trend: 1d close > SMA200 for longs, < SMA200 for shorts
- Regime: Choppiness < 55 (not too choppy)

TARGET: 40-100 total trades over 4 years = 10-25/year. HARD MAX: 150.
Signal size: 0.30 (swing trading = bigger bets, longer holds).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_atr_channel_williams_sma200_v1"
timeframe = "1d"
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

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator"""
    n = len(close)
    williams = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high - lowest_low > 0:
            williams[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return williams

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
    # 1d strategy already uses daily data from primary timeframe
    # For trend, use weekly SMA200 equivalent (smoother)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly SMA100 for longer-term trend (smoother than daily)
    sma_1w = pd.Series(df_1w['close'].values).rolling(window=100, min_periods=100).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # === Local indicators ===
    atr_20 = calculate_atr(high, low, close, period=20)
    williams_14 = calculate_williams_r(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # ATR Channel (20-period = ~20 day channel on 1d)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    channel_mult = 2.5
    channel_upper = sma_20 + channel_mult * atr_20
    channel_lower = sma_20 - channel_mult * atr_20
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 200  # Need enough for ATR(20) + SMA(20) + Williams(14) + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_20[i]) or atr_20[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(williams_14[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(channel_upper[i]) or np.isnan(channel_lower[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (Weekly SMA100) ===
        price_above_weekly_sma = close[i] > sma_1w_aligned[i]
        
        # === REGIME (Choppiness Index) ===
        # Only trade when trending (CHOP < 55)
        is_trending = chop[i] < 55.0
        
        # Skip if too choppy (only when flat)
        if is_trending is False and not in_position:
            signals[i] = 0.0
            continue
        
        # === MOMENTUM (Williams %R) ===
        # -0 to -20 = overbought (momentum fading)
        # -80 to -100 = oversold (momentum building)
        is_overbought = williams_14[i] > -20
        is_oversold = williams_14[i] < -80
        momentum_fading = williams_14[i] > -30  # Not strong momentum
        momentum_building = williams_14[i] < -70  # Strong momentum
        
        # === CHANNEL SIGNALS ===
        current_high = high[i]
        current_low = low[i]
        prev_close = close[i - 1] if i > 0 else close[i]
        
        # Previous bar's channel values
        prev_channel_upper = channel_upper[i - 1] if i > 0 else channel_upper[i]
        prev_channel_lower = channel_lower[i - 1] if i > 0 else channel_lower[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above ATR channel + momentum building ===
            # Price breaks above previous channel high
            # Williams %R showing oversold/momentum building (not overbought)
            breakout_long = current_high > prev_channel_upper
            
            if breakout_long and price_above_weekly_sma:
                # Need momentum confirmation (not fading)
                if momentum_building or vol_spike:
                    desired_signal = SIZE
            
            # === SHORT: Breakdown below ATR channel + momentum fading ===
            # Price breaks below previous channel low
            # Williams %R showing overbought/momentum fading
            breakout_short = current_low < prev_channel_lower
            
            if breakout_short and not price_above_weekly_sma:
                # Need momentum confirmation (not building)
                if momentum_fading or vol_spike:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TIME-BASED EXIT (hold at least 5 bars for swing) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 5:
            # Exit on channel mean reversion
            if position_side > 0 and close[i] < sma_20[i]:
                desired_signal = 0.0
            if position_side < 0 and close[i] > sma_20[i]:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_20[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
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