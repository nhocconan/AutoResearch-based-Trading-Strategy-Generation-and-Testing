#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d regime filter and volume confirmation
# - Primary: 6h price breaks above Camarilla R3 or below S3 from prior 1d session
# - HTF regime: 1d close > 1d EMA(50) for long bias, < EMA(50) for short bias
# - HTF volume: 1d volume > 1.3x 20-period MA for confirmation
# - Entry: Breakout in direction of 1d regime with volume confirmation
# - Exit: Price returns to 1d VWAP or opposite Camarilla level (H4/L4)
# - Position sizing: 0.25
# - Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# - Works in bull/bear: Camarilla levels adapt to volatility, regime filter avoids counter-trend, volume confirms conviction

name = "6h_1d_camarilla_breakout_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA(50) for regime
    ema_1d_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Calculate 1d VWAP
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vwap_1d = (typical_price_1d * volume_1d).cumsum() / volume_1d.cumsum()
    vwap_1d = vwap_1d.replace({np.inf: np.nan, -np.inf: np.nan}).values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate 1d volume MA(20)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_1d_50_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or
            np.isnan(volume_ma_20_1d_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Get prior 1d Camarilla levels (using HTF close of completed 1d bar)
        prior_close = close_1d_aligned[i]  # This is the completed 1d close price
        prior_high = high_1d_aligned[i]
        prior_low = low_1d_aligned[i]
        
        # Calculate Camarilla levels for prior 1d
        range_ = prior_high - prior_low
        if range_ <= 0:
            signals[i] = 0.0
            continue
            
        camarilla_h4 = prior_close + range_ * 1.1 / 2
        camarilla_l4 = prior_close - range_ * 1.1 / 2
        camarilla_h3 = prior_close + range_ * 1.1 / 4
        camarilla_l3 = prior_close - range_ * 1.1 / 4
        
        # Volume confirmation: current 1d volume > 1.3x 20-period MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirm = volume_1d_aligned[i] > 1.3 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > H3 + regime long + volume confirmation
            if (close[i] > camarilla_h3 and 
                prior_close > ema_1d_50_aligned[i] and 
                volume_confirm):
                position = 1
                signals[i] = 0.25
            # Short entry: price < L3 + regime short + volume confirmation
            elif (close[i] < camarilla_l3 and 
                  prior_close < ema_1d_50_aligned[i] and 
                  volume_confirm):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: price returns to VWAP or touches opposite H4/L4 level
            if position == 1:  # Long position
                if (close[i] <= vwap_1d_aligned[i] or 
                    close[i] >= camarilla_h4):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if (close[i] >= vwap_1d_aligned[i] or 
                    close[i] <= camarilla_l4):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals