#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum strategy with 4h MACD trend filter and 1d volume regime filter.
# Uses 4h MACD for trend direction (bullish when MACD > signal line) and 1d volume z-score
# to identify high/low volume regimes. In high volume regimes (z > 0.5), follow 1h momentum
# (price > EMA20). In low volume regimes (z < -0.5), mean revert at 1h VWAP deviations.
# Designed to work in both bull and bear markets by adapting to volume regime and trend.
# Targets 15-35 trades/year with disciplined risk control.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for MACD trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h MACD (12,26,9)
    ema12 = pd.Series(close_4h).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close_4h).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd = ema12 - ema26
    signal_line = pd.Series(macd).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd - signal_line
    
    # Align MACD histogram to 1h timeframe
    macd_hist_aligned = align_htf_to_ltf(prices, df_4h, macd_hist)
    
    # Load 1d data for volume regime filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume z-score (20-day)
    vol_mean = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_std = pd.Series(volume_1d).rolling(window=20, min_periods=20).std().values
    vol_zscore = (volume_1d - vol_mean) / vol_std
    vol_zscore = np.where(vol_std == 0, 0, vol_zscore)
    
    # Align volume z-score to 1h timeframe
    vol_zscore_aligned = align_htf_to_ltf(prices, df_1d, vol_zscore)
    
    # Calculate 1h EMA20 for momentum
    close = prices['close'].values
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1h VWAP for mean reversion
    typical_price = (prices['high'].values + prices['low'].values + prices['close'].values) / 3
    vwap_num = np.cumsum(typical_price * prices['volume'].values)
    vwap_den = np.cumsum(prices['volume'].values)
    vwap = vwap_num / vwap_den
    vwap = np.where(vwap_den == 0, typical_price, vwap)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(macd_hist_aligned[i]) or 
            np.isnan(vol_zscore_aligned[i]) or 
            np.isnan(ema20[i]) or 
            np.isnan(vwap[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol_z = vol_zscore_aligned[i]
        macd_h = macd_hist_aligned[i]
        ema = ema20[i]
        vwap_val = vwap[i]
        
        if position == 0:
            # High volume regime: follow trend
            if vol_z > 0.5:
                if macd_h > 0 and price > ema:
                    signals[i] = 0.20
                    position = 1
                elif macd_h < 0 and price < ema:
                    signals[i] = -0.20
                    position = -1
            # Low volume regime: mean reversion
            elif vol_z < -0.5:
                if price < vwap_val * 0.995:  # 0.5% below VWAP
                    signals[i] = 0.20
                    position = 1
                elif price > vwap_val * 1.005:  # 0.5% above VWAP
                    signals[i] = -0.20
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on trend change or mean reversion
                if vol_z > 0.5:  # high volume regime
                    if macd_h < 0 or price < ema:
                        exit_signal = True
                else:  # low volume regime
                    if price > vwap_val * 1.005:  # reverted to VWAP
                        exit_signal = True
            
            elif position == -1:  # short position
                # Exit on trend change or mean reversion
                if vol_z > 0.5:  # high volume regime
                    if macd_h > 0 or price > ema:
                        exit_signal = True
                else:  # low volume regime
                    if price < vwap_val * 0.995:  # reverted to VWAP
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_MACD_Volume_Regime_Momentum"
timeframe = "1h"
leverage = 1.0