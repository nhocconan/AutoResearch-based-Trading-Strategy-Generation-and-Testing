# 12h_1w_1d_camarilla_breakout_v1
# Uses weekly and daily pivots with volume confirmation and chop regime filter
# Weekly: trend direction (price above/below weekly pivot)
# Daily: entry signals at H3/L3 levels with volume
# Chop filter: avoid ranging markets
# Target: 15-30 trades/year per symbol
name = "12h_1w_1d_camarilla_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w and 1d data
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly pivot for trend direction
    high_w = df_1w['high'].shift(1).values
    low_w = df_1w['low'].shift(1).values
    close_w = df_1w['close'].shift(1).values
    pivot_w = (high_w + low_w + close_w) / 3.0
    weekly_pivot = align_htf_to_ltf(prices, df_1w, pivot_w)
    
    # Daily Camarilla levels
    high_d = df_1d['high'].shift(1).values
    low_d = df_1d['low'].shift(1).values
    close_d = df_1d['close'].shift(1).values
    range_d = high_d - low_d
    h3_d = close_d + range_d * 1.1 / 4
    l3_d = close_d - range_d * 1.1 / 4
    h3_level = align_htf_to_ltf(prices, df_1d, h3_d)
    l3_level = align_htf_to_ltf(prices, df_1d, l3_d)
    
    # Volume confirmation: volume > 1.8 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_confirm = volume > (vol_ma * 1.8)
    
    # Chop regime filter: avoid choppy markets (CHOP > 61.8)
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((highest_high - lowest_low) / (atr * np.sqrt(14))) / np.log10(14)
    chop_filter = chop < 61.8  # trending market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if levels not ready
        if np.isnan(weekly_pivot[i]) or np.isnan(h3_level[i]) or np.isnan(l3_level[i]):
            signals[i] = 0.0
            continue
        
        # Check volume and chop filters
        if not (vol_confirm[i] and chop_filter[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price above weekly pivot and breaks above H3
        if close[i] > weekly_pivot[i] and close[i] > h3_level[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price below weekly pivot and breaks below L3
        elif close[i] < weekly_pivot[i] and close[i] < l3_level[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price crosses weekly pivot in opposite direction
        elif (close[i] < weekly_pivot[i] and position == 1) or (close[i] > weekly_pivot[i] and position == -1):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals