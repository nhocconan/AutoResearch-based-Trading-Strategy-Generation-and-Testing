#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI momentum + Choppiness regime filter
# KAMA adapts to market noise, avoiding whipsaws in sideways markets.
# RSI filters for momentum strength. Choppiness index determines regime:
# CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (follow trend).
# Works in both bull/bear by adapting to regime. Low trade frequency (~15-25/year).
# Uses 1w trend filter for higher timeframe bias.

name = "1d_KAMA_RSI_ChopRegime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - 10-period ER, 2-30 SC
    close_series = pd.Series(close)
    change = abs(close_series.diff(10)).values
    volatility = abs(close_series.diff(1)).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14-period)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Choppiness Index (14-period)
    atr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    atr1 = np.maximum(atr1, np.abs(low[1:] - close[:-1]))
    atr1 = np.concatenate([[np.nan], atr1])
    atr_sum = pd.Series(atr1).rolling(window=14, min_periods=14).sum()
    hh = pd.Series(high).rolling(window=14, min_periods=14).max()
    ll = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    chop = chop.fillna(50).values  # neutral when undefined
    
    # Weekly EMA(34) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(ema_34_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema_trend = ema_34_1w_aligned[i]
        
        # Regime filter: trending when CHOP < 38.2, ranging when CHOP > 61.8
        is_trending = chop_val < 38.2
        is_ranging = chop_val > 61.8
        
        if position == 0:
            # Long: KAMA bullish + RSI momentum + trending regime
            if price > kama_val and rsi_val > 55 and is_trending and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: KAMA bearish + RSI weakness + trending regime
            elif price < kama_val and rsi_val < 45 and is_trending and price < ema_trend:
                signals[i] = -0.25
                position = -1
            # Mean reversion in ranging: buy near support, sell near resistance
            elif is_ranging:
                # Use Bollinger Bands for mean reversion levels
                bb_mid = pd.Series(close).rolling(window=20, min_periods=20).mean()[i]
                bb_std = pd.Series(close).rolling(window=20, min_periods=20).std()[i]
                bb_lower = bb_mid - 2 * bb_std
                bb_upper = bb_mid + 2 * bb_std
                if not np.isnan(bb_lower) and not np.isnan(bb_upper):
                    if price <= bb_lower and rsi_val < 40:  # oversold
                        signals[i] = 0.25
                        position = 1
                    elif price >= bb_upper and rsi_val > 60:  # overbought
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:
            # Exit: KAMA turn bearish OR RSI overbought OR regime change to ranging
            if price < kama_val or rsi_val > 70 or (is_ranging and not is_trending):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: KAMA turn bullish OR RSI oversold OR regime change to ranging
            if price > kama_val or rsi_val < 30 or (is_ranging and not is_trending):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals