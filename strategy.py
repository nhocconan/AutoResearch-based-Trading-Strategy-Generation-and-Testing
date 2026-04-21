#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE for trend, structure, and momentum
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d RSI(14) for momentum filter
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_values = rsi_14.fillna(0).values
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_values)
    
    # 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_34 = pd.Series(atr_14).rolling(window=34, min_periods=34).mean().values
    atr_ratio = atr_14 / atr_ma_34
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # 4h Donchian(20) for entry signals (using daily close as proxy for structure)
    # Since we're on 4h timeframe, we use daily Donchian levels aligned to 4h bars
    donch_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(rsi_14_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_34_1d_aligned[i]
        rsi_val = rsi_14_aligned[i]
        upper_band = donch_high_aligned[i]
        lower_band = donch_low_aligned[i]
        atr_ratio_val = atr_ratio_aligned[i]
        
        if position == 0:
            # Enter long: price above daily Donchian high, uptrend (price > EMA34), 
            # bullish momentum (RSI > 50), moderate volatility
            if (price_close > upper_band and 
                price_close > ema_trend and 
                rsi_val > 50 and 
                atr_ratio_val > 0.6 and atr_ratio_val < 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: price below daily Donchian low, downtrend (price < EMA34), 
            # bearish momentum (RSI < 50), moderate volatility
            elif (price_close < lower_band and 
                  price_close < ema_trend and 
                  rsi_val < 50 and 
                  atr_ratio_val > 0.6 and atr_ratio_val < 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: reverse breakout, trend reversal, or volatility extremes
            if position == 1 and (price_close < lower_band or 
                                  price_close < ema_trend or 
                                  rsi_val < 40 or 
                                  atr_ratio_val > 2.5 or 
                                  atr_ratio_val < 0.4):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > upper_band or 
                                     price_close > ema_trend or 
                                     rsi_val > 60 or 
                                     atr_ratio_val > 2.5 or 
                                     atr_ratio_val < 0.4):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_DailyDonchian_EMA34_RSI_Filter"
timeframe = "4h"
leverage = 1.0