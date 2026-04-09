#!/usr/bin/env python3
# 1d_kama_rsi_chop_filter_v1
# Hypothesis: 1d strategy using KAMA for trend direction, RSI(14) for momentum confirmation, and Choppiness Index regime filter (CHOP>61.8 = ranging for mean reversion, CHOP<38.2 = trending for trend following). Uses 1w HTF EMA(34) for higher timeframe alignment. Discrete position sizing (0.25) to minimize fee churn. Target: 7-25 trades/year (30-100 total over 4 years). Works in bull/bear: KAMA adapts to volatility, RSI avoids exhaustion, chop filter ensures regime-appropriate logic, HTF EMA prevents counter-trend trades.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA (Adaptive Moving Average)
    close_s = pd.Series(close)
    direction = np.abs(close_s.diff(10).values)
    volatility = np.abs(close_s.diff(1)).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Choppiness Index (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_sum = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).sum().values
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    denominator = np.log10(atr_period) * (highest_high - lowest_low)
    chop = 100 * np.log10(tr_sum / denominator)
    chop = np.where(denominator == 0, np.nan, chop)
    
    # Multi-timeframe: 1w EMA(34) trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    close_1w_s = pd.Series(close_1w)
    ema_34_1w = close_1w_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi_values[i]) or np.isnan(chop[i]) or
            np.isnan(close[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters
        ranging_market = chop[i] > 61.8
        trending_market = chop[i] < 38.2
        # HTF trend filter
        htf_uptrend = close[i] > ema_34_1w_aligned[i]
        htf_downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or trend reversal
            if rsi_values[i] > 70 or (close[i] < kama[i] and trending_market):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or trend reversal
            if rsi_values[i] < 30 or (close[i] > kama[i] and trending_market):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion in ranging markets: fade extremes
            if ranging_market:
                if close[i] < kama[i] and rsi_values[i] < 30:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > kama[i] and rsi_values[i] > 70:
                    position = -1
                    signals[i] = -0.25
            # Trend following in trending markets: pullback to KAMA
            elif trending_market:
                if close[i] > kama[i] and rsi_values[i] > 50 and htf_uptrend:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < kama[i] and rsi_values[i] < 50 and htf_downtrend:
                    position = -1
                    signals[i] = -0.25
    
    return signals