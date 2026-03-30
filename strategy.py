#!/usr/bin/env python3
"""
Experiment #007: 6h Ichimoku Cloud Regime + Elder Ray Momentum

HYPOTHESIS: Novel combination not tried in prior experiments:
1. 1d Ichimoku cloud direction (TIER 8 indicator) - proven regime filter
2. Elder Ray (Bull Power) for momentum confirmation
3. Volume spike 2.0x as entry confirmation
4. Choppiness filter <50 to avoid range markets

WHY IT SHOULD WORK:
- Ichimoku cloud provides robust HTF trend direction (KUMO is structural support/resistance)
- Elder Ray measures actual buying/selling pressure behind price
- Works in both bull (trend following) and bear (cloud breakdown = short)
- 6h timeframe = 12 bars/day = reasonable trade frequency

TARGET: 75-200 total trades over 4 years (18-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_ichimoku_elder_ray_1d_vol_v1"
timeframe = "6h"
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

def calculate_ichimoku_cloud(high, low, close, period=9):
    """
    Ichimoku Cloud - calculate Tenkan-sen, Kijun-sen, and Cloud (Kumo)
    Returns: tenkan, kijun, senkou_a, senkou_b
    Cloud (KUMO) = area between Senkou Span A and B
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan = np.zeros(n, dtype=np.float64)
    kijun = np.zeros(n, dtype=np.float64)
    
    for i in range(period - 1, n):
        high_9 = np.max(high[i - period + 1:i + 1])
        low_9 = np.min(low[i - period + 1:i + 1])
        tenkan[i] = (high_9 + low_9) / 2
        
        high_26 = np.max(high[max(0, i - period + 1):i + 1]) if i >= period else high_9
        low_26 = np.min(low[max(0, i - period + 1):i + 1]) if i >= period else low_9
        kijun[i] = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted forward 26 periods
    senkou_a = np.zeros(n, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            senkou_a[i] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted forward 26
    senkou_b = np.full(n, np.nan)
    period_b = 52
    for i in range(period_b - 1, n):
        high_52 = np.max(high[i - period_b + 1:i + 1])
        low_52 = np.min(low[i - period_b + 1:i + 1])
        senkou_b[i] = (high_52 + low_52) / 2
    
    return tenkan, kijun, senkou_a, senkou_b

def calculate_elder_ray(high, low, close, ema_period=13):
    """
    Elder Ray (Bull Power / Bear Power)
    Bull Power = High - EMA(13)
    Bear Power = Low - EMA(13)
    Bull Power > 0 + rising = bullish momentum
    Bear Power < 0 + falling = bearish momentum
    """
    n = len(close)
    if n < ema_period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # EMA of close
    ema = pd.Series(close).ewm(span=ema_period, min_periods=ema_period, adjust=False).mean().values
    
    bull_power = high - ema
    bear_power = low - ema
    
    return bull_power, bear_power

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging - DON'T enter
    CHOP < 50 = trending - GOOD to enter
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest > lowest and atr_sum > 0:
            range_hl = highest - lowest
            chop[i] = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load 1d data ONCE for HTF Ichimoku ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Ichimoku cloud
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku_cloud(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=9
    )
    
    # Align to 6h (shift by 1 to avoid look-ahead)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # === Local 6h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    bull_power, bear_power = calculate_elder_ray(high, low, close, ema_period=13)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # EMA for Elder Ray baseline
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    # Warmup: 52 for Senkou B + 26 for cloud shift + max(local periods)
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === HTF ICHIMOKU CLOUD REGIME ===
        # Cloud formed by Senkou A vs Senkou B
        cloud_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Bullish cloud: price above cloud + cloud rising
        price_above_cloud = close[i] > cloud_top
        cloud_bullish = senkou_a_aligned[i] > senkou_b_aligned[i]  # Green cloud
        
        # Bearish cloud: price below cloud + cloud falling
        price_below_cloud = close[i] < cloud_bottom
        cloud_bearish = senkou_a_aligned[i] < senkou_b_aligned[i]  # Red cloud
        
        # TK cross on 1d (tenkan crosses kijun) - additional confirmation
        tk_bullish_cross = tenkan_aligned[i] > kijun_aligned[i]
        tk_bearish_cross = tenkan_aligned[i] < kijun_aligned[i]
        
        # === CHOPPINESS FILTER ===
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 50
        
        # === ELDER RAY MOMENTUM ===
        # Bull power positive = buying pressure
        # Bear power negative = selling pressure
        bull_momentum = bull_power[i] > 0
        bear_momentum = bear_power[i] < 0
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 2.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Bullish cloud + TK cross up + price above cloud + volume spike + trending ===
            long_conditions = (
                cloud_bullish and 
                tk_bullish_cross and 
                price_above_cloud and 
                vol_spike and 
                is_trending and
                bull_momentum
            )
            if long_conditions:
                desired_signal = SIZE
            
            # === SHORT: Bearish cloud + TK cross down + price below cloud + volume spike + trending ===
            short_conditions = (
                cloud_bearish and 
                tk_bearish_cross and 
                price_below_cloud and 
                vol_spike and 
                is_trending and
                bear_momentum
            )
            if short_conditions:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Long: exit if price drops 2.5 ATR from entry
                stop_price = entry_price - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if cloud turns bearish
                if cloud_bearish:
                    desired_signal = 0.0
                
                # Exit if choppy
                if is_choppy:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Short: exit if price rises 2.5 ATR from entry
                stop_price = entry_price + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if cloud turns bullish
                if cloud_bullish:
                    desired_signal = 0.0
                
                # Exit if choppy
                if is_choppy:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 6 bars (1.5 days on 6h) to avoid fee churn ===
        if in_position and (i - entry_bar) < 6:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals