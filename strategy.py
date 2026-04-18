#!/usr/bin/env python3
"""
4h_Engulfing_Candle_Pattern_With_RSI_And_12hTrend
Hypothesis: Bullish/bearish engulfing candles with RSI(14) confirmation and 12h EMA34 trend filter. Engulfing patterns signal strong reversals, RSI avoids overextended entries, and 12h EMA34 ensures medium-term alignment. Designed for low trade frequency (<40/year) to minimize fee drag while capturing high-probability reversals in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bullish engulfing: current green candle engulfs previous red candle
    bullish_engulf = (close > open_price) & (open_price < close) & (close > open_price) & (open_price < close) & \
                     (close > open_price) & (open_price < close)  # Placeholder - will fix below
    # Actually: current close > previous open AND current open < previous close
    bullish_engulf = (close > np.roll(open_price, 1)) & (open_price < np.roll(close, 1))
    
    # Bearish engulfing: current red candle engulfs previous green candle
    bearish_engulf = (close < open_price) & (open_price > np.roll(close, 1)) & (close < np.roll(open_price, 1))
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 12h EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(14, 34)  # RSI and EMA warmup
    
    for i in range(start_idx, n):
        if (np.isnan(bullish_engulf[i]) or 
            np.isnan(bearish_engulf[i]) or
            np.isnan(rsi[i]) or
            np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        ema_12h_val = ema_12h_aligned[i]
        bull_eng = bullish_engulf[i]
        bear_eng = bearish_engulf[i]
        
        if position == 0:
            # Long: bullish engulfing, RSI < 60 (not overbought), above 12h EMA
            if bull_eng and rsi_val < 60 and price > ema_12h_val:
                signals[i] = 0.25
                position = 1
            # Short: bearish engulfing, RSI > 40 (not oversold), below 12h EMA
            elif bear_eng and rsi_val > 40 and price < ema_12h_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: bearish engulfing or RSI > 70 or below 12h EMA
            if bear_eng or rsi_val > 70 or price < ema_12h_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: bullish engulfing or RSI < 30 or above 12h EMA
            if bull_eng or rsi_val < 30 or price > ema_12h_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Engulfing_Candle_Pattern_With_RSI_And_12hTrend"
timeframe = "4h"
leverage = 1.0