#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and ATR volatility filter
# - Primary: 12h price breaks above/below Camarilla H3/L3 levels (strong support/resistance)
# - Volume filter: 1d volume > 1.3x 20-period volume MA to confirm breakout with participation
# - Volatility filter: ATR(14) < 0.6 * 50-period ATR MA to avoid extreme volatility whipsaws
# - Exit: Price returns to Camarilla H4/L4 levels
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Camarilla levels adapt to volatility, volume confirms institutional interest,
#   volatility filter avoids choppy markets, effective in both bull and bear markets

name = "12h_1d_camarilla_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Camarilla pivot levels for 12h timeframe (based on previous bar)
    # Camarilla: H4 = close + 1.1*(high-low)/2, H3 = close + 1.1*(high-low)/4
    #            L3 = close - 1.1*(high-low)/4, L4 = close - 1.1*(high-low)/2
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_range = prev_high - prev_low
    camarilla_h3 = prev_close + 1.1 * camarilla_range / 4
    camarilla_l3 = prev_close - 1.1 * camarilla_range / 4
    camarilla_h4 = prev_close + 1.1 * camarilla_range / 2
    camarilla_l4 = prev_close - 1.1 * camarilla_range / 2
    
    # Calculate 1d volume confirmation: volume > 1.3x 20-period volume MA
    volume_ma_20_1d = pd.Series(volume_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Calculate 50-period ATR MA for volatility regime filter
    atr_ma_50 = pd.Series(atr).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or
            np.isnan(volume_ma_20_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(atr_ma_50[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.3x 20-period MA
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_1d_current[i] > 1.3 * volume_ma_20_1d_aligned[i]
        
        # Volatility filter: ATR(14) < 0.6 * 50-period ATR MA (avoid extreme volatility)
        vol_filter = atr[i] < 0.6 * atr_ma_50[i]
        
        if position == 0:  # Flat - look for new Camarilla breakouts
            # Long entry: Price breaks above H3 + vol confirmation + vol filter
            if close[i] > camarilla_h3[i] and vol_confirm and vol_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below L3 + vol confirmation + vol filter
            elif close[i] < camarilla_l3[i] and vol_confirm and vol_filter:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to H4/L4 levels
            if position == 1:  # Long position
                if close[i] <= camarilla_h4[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] >= camarilla_l4[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals