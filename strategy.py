# 140932: 12h Camarilla R1/S1 Breakout with 1d Trend and Volume Confirmation
# Hypothesis: Breakouts at daily Camarilla R1/S1 levels with 1d trend alignment and volume confirmation.
# Uses 1d EMA(34) for trend and 20-period volume spike for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull (breakout continuation) and bear (mean reversion at R1/S1) via trend filter.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla levels (R1, S1)
    range_ = prev_high - prev_low
    R1 = prev_close + (range_ * 1.1 / 12)
    S1 = prev_close - (range_ * 1.1 / 12)
    
    # 1d EMA(34) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = ema_34_1d[1:] > ema_34_1d[:-1]  # Rising EMA = uptrend
    trend_up = np.concatenate([[False], trend_up])  # Align with 1d index
    
    # Volume confirmation: 20-period volume spike (2.0x EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    # Align 1d indicators to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(trend_up_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Break above R1 in uptrend OR bounce at S1 in downtrend
            if (trend_up_aligned[i] > 0.5 and  # 1d uptrend
                close[i] >= R1_aligned[i] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            elif (trend_up_aligned[i] <= 0.5 and  # 1d downtrend
                  close[i] <= S1_aligned[i] and
                  vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Break below S1 in downtrend OR bounce at R1 in uptrend
            elif (trend_up_aligned[i] <= 0.5 and  # 1d downtrend
                  close[i] <= S1_aligned[i] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            elif (trend_up_aligned[i] > 0.5 and  # 1d uptrend
                  close[i] >= R1_aligned[i] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Reverse signal or opposite touch
            if (trend_up_aligned[i] <= 0.5 and  # 1d downtrend
                close[i] <= S1_aligned[i]):  # Touch S1
                signals[i] = 0.0
                position = 0
            elif (trend_up_aligned[i] > 0.5 and  # 1d uptrend
                  close[i] >= R1_aligned[i]):  # Touch R1
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Reverse signal or opposite touch
            if (trend_up_aligned[i] > 0.5 and  # 1d uptrend
                close[i] >= R1_aligned[i]):  # Touch R1
                signals[i] = 0.0
                position = 0
            elif (trend_up_aligned[i] <= 0.5 and  # 1d downtrend
                  close[i] <= S1_aligned[i]):  # Touch S1
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals