# 12h_Camarilla_Pivot_Reversal_1dTrend_Volume
# Reversal strategy using 1d Camarilla pivot levels (H4/L4) with 1d trend filter and volume confirmation.
# Long when price touches L4 (support) during 1d uptrend with volume spike; short when touches H4 (resistance) during 1d downtrend.
# Uses 12h timeframe to reduce trade frequency (target: 25-40 trades/year) and avoid fee drag.
# Works in both bull and bear markets by following 1d trend direction for entries.

name = "12h_Camarilla_Pivot_Reversal_1dTrend_Volume"
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
    
    # Daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla H4 and L4 levels from previous day's OHLC
    # Camarilla: H4 = close + 1.5*(high-low)/2, L4 = close - 1.5*(high-low)/2
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    H4 = prev_close + 1.5 * (prev_high - prev_low) / 2
    L4 = prev_close - 1.5 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 12h timeframe (constant throughout the day)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d EMA34 direction
    ema34_rising = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1d_aligned[1:] > ema34_1d_aligned[:-1]
    ema34_falling[1:] = ema34_1d_aligned[1:] < ema34_1d_aligned[:-1]
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Sufficient warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_rising[i]) or 
            np.isnan(ema34_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price touches L4 (support), 1d EMA34 rising, volume filter
            long_cond = (low[i] <= L4_aligned[i]) and ema34_rising[i] and volume_filter[i]
            # Short conditions: price touches H4 (resistance), 1d EMA34 falling, volume filter
            short_cond = (high[i] >= H4_aligned[i]) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses above H4 (resistance) or closes below L4
            if high[i] >= H4_aligned[i] or close[i] < L4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses below L4 (support) or closes above H4
            if low[i] <= L4_aligned[i] or close[i] > H4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals