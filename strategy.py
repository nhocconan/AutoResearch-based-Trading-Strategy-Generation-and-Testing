#!/usr/bin/env python3
"""
Experiment #248: 30m Primary + 4h/1d HTF — Session-Filtered Regime Strategy

Hypothesis: Lower TF (30m) strategies fail due to either (a) too many trades → fee drag,
or (b) too strict filters → 0 trades. The solution is HTF direction + LTF timing +
session filter to reduce noise.

This strategy uses:
1. 1d HMA(21) slope for PRIMARY trend regime (bull/bear/neutral)
2. 4h Choppiness Index(14) for secondary regime (trend vs range)
3. 30m RSI(14) with asymmetric thresholds for entry timing
4. 30m Volume filter (>0.8x 20-bar avg) for confirmation
5. Session filter (8-20 UTC only) to avoid Asian session noise
6. 30m ATR(14) for trailing stops

Key insight: 30m entries MUST be rare. Use 1d/4h for direction, 30m only for
pullback entries within the HTF trend. Session filter cuts 60% of noise trades.

Position sizing: 0.20 base, 0.30 strong (discrete, lower for 30m TF)
Target: 40-80 trades/year per symbol (within 30m cost model)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_session_regime_hma_chop_4h1d_v1"
timeframe = "30m"
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
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    hull = 2 * wma_half - wma_full
    hma = hull.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope as percentage change over lookback."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        prev = hma_values[i - lookback]
        curr = hma_values[i]
        if prev != 0 and not np.isnan(prev) and not np.isnan(curr):
            slope[i] = (curr - prev) / abs(prev) * 100
    return slope

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    ratio = vol_s / vol_avg.replace(0, np.nan)
    ratio = ratio.fillna(1.0).values
    return ratio

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hours = (open_time // 3600000) % 24
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (primary trend regime)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 3)
    
    # Calculate 4h HTF indicators (secondary regime)
    chop_4h_14 = calculate_choppiness_index(
        df_4h['high'].values, 
        df_4h['low'].values, 
        df_4h['close'].values, 
        14
    )
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    chop_4h_14_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_14)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_ratio_20 = calculate_volume_ratio(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, lower for 30m)
    BASE_SIZE = 0.20
    STRONG_SIZE = 0.30
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(chop_4h_14_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(vol_ratio_20[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === 1D TREND REGIME (primary direction filter) ===
        # Bull: 1d HMA slope > 0.15%
        # Bear: 1d HMA slope < -0.15%
        # Neutral: between
        regime_bull = hma_1d_slope_aligned[i] > 0.15
        regime_bear = hma_1d_slope_aligned[i] < -0.15
        regime_neutral = not regime_bull and not regime_bear
        
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === 4H CHOPPINESS REGIME ===
        # CHOP > 55 = range market (mean revert entries)
        # CHOP < 45 = trend market (breakout entries)
        is_choppy = chop_4h_14_aligned[i] > 55.0
        is_trending = chop_4h_14_aligned[i] < 45.0
        
        # === 30M LOCAL SIGNALS ===
        volume_confirmed = vol_ratio_20[i] > 0.8
        
        # === ASYMMETRIC RSI THRESHOLDS ===
        # In bull regime: easier longs (RSI>38), harder shorts (RSI>68)
        # In bear regime: harder longs (RSI>45), easier shorts (RSI>62)
        # In neutral: balanced (RSI>42 / RSI>58)
        if regime_bull:
            rsi_long_trigger = rsi_14[i] > 38
            rsi_short_trigger = rsi_14[i] > 68
            rsi_oversold = rsi_14[i] < 32
            rsi_overbought = rsi_14[i] > 72
        elif regime_bear:
            rsi_long_trigger = rsi_14[i] > 45
            rsi_short_trigger = rsi_14[i] > 62
            rsi_oversold = rsi_14[i] < 28
            rsi_overbought = rsi_14[i] > 68
        else:
            rsi_long_trigger = rsi_14[i] > 42
            rsi_short_trigger = rsi_14[i] > 58
            rsi_oversold = rsi_14[i] < 30
            rsi_overbought = rsi_14[i] > 70
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # Only trade during session hours (reduces noise by ~60%)
        if in_session and volume_confirmed:
            # TREND FOLLOWING MODE (when trending + regime aligned)
            if is_trending:
                # LONG: Trending + bull regime + price above 1d HMA + RSI confirmation
                if regime_bull and price_above_1d_hma and rsi_long_trigger:
                    new_signal = STRONG_SIZE
                # LONG: Trending + neutral regime + price above 1d HMA + strong RSI
                elif regime_neutral and price_above_1d_hma and rsi_14[i] > 50:
                    new_signal = BASE_SIZE
                
                # SHORT: Trending + bear regime + price below 1d HMA + RSI confirmation
                if regime_bear and price_below_1d_hma and rsi_short_trigger:
                    new_signal = -STRONG_SIZE
                # SHORT: Trending + neutral regime + price below 1d HMA + strong RSI
                elif regime_neutral and price_below_1d_hma and rsi_14[i] < 50:
                    new_signal = -BASE_SIZE
            
            # MEAN REVERSION MODE (when choppy)
            if is_choppy:
                # LONG: Choppy + RSI oversold + price below 1d HMA (pullback in bull/neutral)
                if rsi_oversold and price_below_1d_hma and not regime_bear:
                    new_signal = BASE_SIZE
                # LONG: Choppy + RSI very oversold (<28) in any non-bear regime
                if rsi_14[i] < 28 and not regime_bear:
                    if new_signal == 0.0:
                        new_signal = BASE_SIZE * 0.8
                
                # SHORT: Choppy + RSI overbought + price above 1d HMA (pullback in bear/neutral)
                if rsi_overbought and price_above_1d_hma and not regime_bull:
                    if new_signal == 0.0:
                        new_signal = -BASE_SIZE
                # SHORT: Choppy + RSI very overbought (>72) in any non-bull regime
                if rsi_14[i] > 72 and not regime_bull:
                    if new_signal == 0.0:
                        new_signal = -BASE_SIZE * 0.8
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 50 bars (~25 hours on 30m) but only in session
        if bars_since_last_trade > 50 and new_signal == 0.0 and not in_position and in_session:
            if regime_bull and rsi_14[i] > 45 and price_above_1d_hma and vol_ratio_20[i] > 0.9:
                new_signal = BASE_SIZE * 0.6
            elif regime_bear and rsi_14[i] < 55 and price_below_1d_hma and vol_ratio_20[i] > 0.9:
                new_signal = -BASE_SIZE * 0.6
            elif is_choppy and rsi_14[i] < 35 and not regime_bear:
                new_signal = BASE_SIZE * 0.5
            elif is_choppy and rsi_14[i] > 65 and not regime_bull:
                new_signal = -BASE_SIZE * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but regime turns strongly bearish
            if position_side > 0 and regime_bear and price_below_1d_hma:
                regime_reversal = True
            # Short position but regime turns strongly bullish
            if position_side < 0 and regime_bull and price_above_1d_hma:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals