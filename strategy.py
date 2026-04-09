#!/usr/bin/env python3
# 1d_kama_rsi_chop_regime_v1
# Hypothesis: 1d strategy using KAMA(10) for trend direction, RSI(14) for momentum confirmation, and Choppiness Index regime filter (CHOP > 61.8 = ranging for mean reversion, CHOP < 38.2 = trending for trend following). Uses 1w HTF EMA(50) for higher timeframe alignment. Long when: price > KAMA, RSI > 50, CHOP < 38.2 (trending), and price > 1w EMA(50). Short when: price < KAMA, RSI < 50, CHOP < 38.2 (trending), and price < 1w EMA(50). In ranging markets (CHOP > 61.8), fade extremes: long when RSI < 30 and price < KAMA, short when RSI > 70 and price > KAMA. Discrete position sizing (0.25) to minimize fee churn. Target: 7-25 trades/year (30-100 total over 4 years). Works in bull/bear: KAMA adapts to trend speed, RSI confirms momentum, chop filter selects appropriate regime, HTF EMA ensures alignment with higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_regime_v1"
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
    
    # KAMA (10-period)
    close_s = pd.Series(close)
    direction = np.abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(window=10, min_periods=10).sum()
    er = direction / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14-period)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Choppiness Index regime filter (14-period)
    atr_period = 14
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr_series = pd.Series(tr)
    atr_series = tr_series.rolling(window=atr_period, min_periods=atr_period).mean()
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    atr_sum = tr_series.rolling(window=atr_period, min_periods=atr_period).sum().values
    denominator = np.log10(atr_period) * (highest_high - lowest_low)
    denominator = np.where(denominator == 0, np.nan, denominator)
    chop = 100 * np.log10(atr_sum / denominator)
    
    # Multi-timeframe: 1w EMA(50) trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    close_1w_s = pd.Series(close_1w)
    ema_50_1w = close_1w_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(close[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters
        trending_market = chop[i] < 38.2
        ranging_market = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit conditions
            if trending_market:
                # Exit trend-following long: price < KAMA or RSI < 40
                if close[i] < kama[i] or rsi[i] < 40:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_market:
                # Exit mean-reversion long: RSI > 50
                if rsi[i] > 50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:
                # Neutral chop: exit
                position = 0
                signals[i] = 0.0
                
        elif position == -1:  # Short position
            # Exit conditions
            if trending_market:
                # Exit trend-following short: price > KAMA or RSI > 60
                if close[i] > kama[i] or rsi[i] > 60:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_market:
                # Exit mean-reversion short: RSI < 50
                if rsi[i] < 50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:
                # Neutral chop: exit
                position = 0
                signals[i] = 0.0
        else:  # Flat
            # Check for entry conditions based on regime
            if trending_market:
                # Trend-following entries
                bullish_entry = (close[i] > kama[i]) and (rsi[i] > 50) and (close[i] > ema_50_1w_aligned[i])
                bearish_entry = (close[i] < kama[i]) and (rsi[i] < 50) and (close[i] < ema_50_1w_aligned[i])
                
                if bullish_entry:
                    position = 1
                    signals[i] = 0.25
                elif bearish_entry:
                    position = -1
                    signals[i] = -0.25
            elif ranging_market:
                # Mean-reversion entries (fade extremes)
                bullish_entry = (rsi[i] < 30) and (close[i] < kama[i])
                bearish_entry = (rsi[i] > 70) and (close[i] > kama[i])
                
                if bullish_entry:
                    position = 1
                    signals[i] = 0.25
                elif bearish_entry:
                    position = -1
                    signals[i] = -0.25
            # Neutral chop: no entries
    
    return signals