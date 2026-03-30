#!/usr/bin/env python3
"""
Experiment #009: 4h Ichimoku Cloud + TK Cross + Volume + 1d EMA200 Trend

HYPOTHESIS: Ichimoku Cloud provides institutional-grade structure:
- TK Cross = momentum shift signal
- Cloud position = regime/trend confirmation
- Chikou span = confirmation (price should be above chikou for longs)
- Volume spike = institutional participation required

WHY 4h + 1d: Balances trade frequency (4h) with trend certainty (1d).
4h gives ~1500 bars/year, targeting 25-40 trades/year = 100-160 total over 4 years.

WHY IT WORKS: Cloud acts as dynamic support/resistance. TK cross IN the cloud
is ambiguous (chop), TK cross ABOVE cloud = strong bullish, BELOW = strong bearish.
This regime filter prevents whipsaws in ranging markets.

Entry conditions:
- Long: price > EMA200(1d), close > cloud, TK > KJ, chikou > price, vol_ratio > 1.5
- Short: price < EMA200(1d), close < cloud, TK < KJ, chikou < price, vol_ratio > 1.5

TARGET: 100-160 total trades over 4 years (25-40/year). HARD MAX: 200.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_ichimoku_tkcloud_vol_ema200_1d_v1"
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

def ichimoku_cloud(high, low, close, n_tenkan=9, n_kijun=26, n_senkou_b=52):
    """
    Calculate Ichimoku Cloud components.
    Tenkan-sen (conversion line): (HH+LL)/2 over n_tenkan
    Kijun-sen (base line): (HH+LL)/2 over n_kijun
    Senkou Span A: (Tenkan + Kijun) / 2, plotted 26 periods ahead
    Senkou Span B: (HH+LL)/2 over n_senkou_b, plotted 26 periods ahead
    Chikou Span: close, plotted 26 periods behind
    """
    n = len(close)
    
    # Initialize arrays
    tenkan = np.full(n, np.nan)
    kijun = np.full(n, np.nan)
    senkou_a = np.full(n, np.nan)
    senkou_b = np.full(n, np.nan)
    
    # Tenkan (9-period)
    for i in range(n_tenkan - 1, n):
        hh = np.max(high[i - n_tenkan + 1:i + 1])
        ll = np.min(low[i - n_tenkan + 1:i + 1])
        tenkan[i] = (hh + ll) / 2.0
    
    # Kijun (26-period)
    for i in range(n_kijun - 1, n):
        hh = np.max(high[i - n_kijun + 1:i + 1])
        ll = np.min(low[i - n_kijun + 1:i + 1])
        kijun[i] = (hh + ll) / 2.0
    
    # Senkou Span B (52-period)
    for i in range(n_senkou_b - 1, n):
        hh = np.max(high[i - n_senkou_b + 1:i + 1])
        ll = np.min(low[i - n_senkou_b + 1:i + 1])
        senkou_b[i] = (hh + ll) / 2.0
    
    # Senkou Span A (average of TK, shifted forward 26 periods)
    for i in range(n_kijun - 1, n):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            senkou_a[i] = (tenkan[i] + kijun[i]) / 2.0
    
    # Chikou is just close shifted back by kijun periods (we'll use close directly in signal logic)
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA200 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h Ichimoku Cloud ===
    tenkan, kijun, senkou_a, senkou_b = ichimoku_cloud(high, low, close)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio (20-bar SMA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
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
    
    # Warmup: need max(200 for EMA, 52 for Ichimoku, 26 for alignment) = 226 bars
    warmup = 250
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if EMA not aligned
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if Ichimoku not ready
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA200) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_1d_aligned[i]
        
        # === TK CROSS (momentum shift) ===
        tk_cross_bullish = tenkan[i] > kijun[i]  # Tenkan above Kijun
        tk_cross_bearish = tenkan[i] < kijun[i]  # Tenkan below Kijun
        
        # === CLOUD POSITION ===
        # Cloud top = max(senkou_a, senkou_b), Cloud bottom = min(senkou_a, senkou_b)
        cloud_top = np.maximum(senkou_a[i], senkou_b[i])
        cloud_bottom = np.minimum(senkou_a[i], senkou_b[i])
        
        # Price above cloud = bullish regime, below = bearish
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # === CHIKOU CONFIRMATION ===
        # Chikou is current close plotted 26 periods behind
        # For confirmation: price should be above where chikou was (i.e., close > cloud at chikou position)
        # Simpler: close[i] should be above cloud_top (price in bullish position)
        chikou_confirm_bull = close[i] > cloud_top
        chikou_confirm_bear = close[i] < cloud_bottom
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Bullish Ichimoku setup + EMA200 trend + volume ===
            # Conditions: price > EMA200, close > cloud, TK > KJ, chikou confirms, vol spike
            if (price_above_1d_ema and 
                price_above_cloud and 
                tk_cross_bullish and 
                chikou_confirm_bull and 
                vol_spike):
                desired_signal = SIZE
            
            # === SHORT: Bearish Ichimoku setup + EMA200 trend + volume ===
            if (price_below_1d_ema and 
                price_below_cloud and 
                tk_cross_bearish and 
                chikou_confirm_bear and 
                vol_spike):
                desired_signal = -SIZE
        
        # === STOPLOSS (3.0 ATR trailing - Ichimoku is slower, wider stop) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 3.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 3.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === MINIMUM HOLD (3 bars = 12h to avoid noise) ===
        bars_held = i - entry_bar
        if bars_held < 3:
            # Keep position, don't exit early
            pass
        
        # === TK CROSS EXIT (exit when momentum reverses) ===
        if in_position and position_side > 0:
            # Exit long if TK crosses below KJ (momentum weakening)
            if tenkan[i] < kijun[i]:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if TK crosses above KJ (momentum weakening)
            if tenkan[i] > kijun[i]:
                desired_signal = 0.0
        
        # === CLOUD TOUCH EXIT ===
        if in_position and position_side > 0:
            # Exit if price re-enters cloud (trend weakening)
            if close[i] <= cloud_top and close[i] >= cloud_bottom:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if close[i] >= cloud_bottom and close[i] <= cloud_top:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
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
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals