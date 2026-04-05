# 2025-06-03
# Experiment #8139: 6-hour Ichimoku Cloud with 12h trend filter and volume confirmation
# Hypothesis: Ichimoku cloud provides dynamic support/resistance. Entry occurs when:
#   - Price crosses above/below Tenkan/Kijun lines (TK cross)
#   - Price is above/below cloud (Kumo) from 12h timeframe for trend filter
#   - Volume > 2x 20-period MA for confirmation
#   - Tenkan > Kijun for long, Tenkan < Kijun for short
# Uses 6h primary timeframe with 12h Ichimoku for trend context.
# Targets 50-150 trades over 4 years by requiring multiple confluence factors.
# Works in bull/bear markets via trend filter and cloud support/resistance.

#!/usr/bin/env python3
from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8139_6h_ichimoku12h_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TK_PERIOD = 9          # Tenkan-sen period
KJ_PERIOD = 26         # Kijun-sen period
SSA_PERIOD = 52        # Senkou Span A period
SSB_PERIOD = 26        # Senkou Span B period
DISPLACEMENT = 26      # Kumo displacement
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan = (pd.Series(high).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).max() + 
              pd.Series(low).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = (pd.Series(high).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).max() + 
             pd.Series(low).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2).shift(DISPLACEMENT)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high).rolling(window=SSA_PERIOD, min_periods=SSA_PERIOD).max() + 
                 pd.Series(low).rolling(window=SSB_PERIOD, min_periods=SSB_PERIOD).min()) / 2).shift(DISPLACEMENT)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Ichimoku
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    tenkan_12h, kijun_12h, senkou_a_12h, senkou_b_12h = calculate_ichimoku(high_12h, low_12h, close_12h)
    
    # Determine cloud top and bottom
    cloud_top_12h = np.maximum(senkou_a_12h, senkou_b_12h)
    cloud_bottom_12h = np.minimum(senkou_a_12h, senkou_b_12h)
    
    # Price above/below cloud: 1 = above (bullish), -1 = below (bearish), 0 = in cloud
    price_vs_cloud = np.where(close_12h > cloud_top_12h, 1, 
                             np.where(close_12h < cloud_bottom_12h, -1, 0))
    price_vs_cloud_aligned = align_htf_to_ltf(prices, df_12h, price_vs_cloud)
    
    # Tenkan > Kijun for bullish momentum, < for bearish
    tk_bullish = tenkan_12h > kijun_12h
    tk_bearish = tenkan_12h < kijun_12h
    tk_bullish_aligned = align_htf_to_ltf(prices, df_12h, tk_bullish.astype(int))
    tk_bearish_aligned = align_htf_to_ltf(prices, df_12h, tk_bearish.astype(int))
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Ichimoku on 6h for entry signals
    tenkan_6h, kijun_6h, _, _ = calculate_ichimoku(high, low, close)
    
    # TK cross signals
    tk_cross_up = (tenkan_6h > kijun_6h) & (tenkan_6h <= kijun_6h)  # crossed up
    tk_cross_down = (tenkan_6h < kijun_6h) & (tenkan_6h >= kijun_6h)  # crossed down
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TK_PERIOD, KJ_PERIOD, SSA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + DISPLACEMENT + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_cloud_aligned[i]) or np.isnan(tk_bullish_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine trend bias from 12h Ichimoku
        bull_trend = price_vs_cloud_aligned[i] == 1      # price above cloud
        bear_trend = price_vs_cloud_aligned[i] == -1     # price below cloud
        bull_momentum = tk_bullish_aligned[i] == 1       # Tenkan > Kijun
        bear_momentum = tk_bearish_aligned[i] == 1       # Tenkan < Kijun
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions - require TK cross in direction of trend
        long_entry = bull_trend and bull_momentum and tk_cross_up[i] and volume_confirmed
        short_entry = bear_trend and bear_momentum and tk_cross_down[i] and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals