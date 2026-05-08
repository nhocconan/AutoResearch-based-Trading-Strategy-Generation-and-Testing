# 12h_Camarilla_Pivot_Breakout_1wTrend_VolumeFilter
# Strategy: 12h timeframe using weekly Camarilla pivot levels with trend filter and volume confirmation
# Hypothesis: Weekly pivot levels provide strong support/resistance that work in both bull and bear markets
# Uses 1w EMA50 for trend alignment and volume spike >1.8 to confirm breakouts
# Target: 15-25 trades/year to minimize fee drag while capturing significant moves
# Timeframe: 12h (lower frequency reduces overtrading risk)

name = "12h_Camarilla_Pivot_Breakout_1wTrend_VolumeFilter"
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
    
    # Get weekly data for trend filter and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate weekly Camarilla levels from previous week
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    
    # Align to 12h
    prev_close_aligned = align_htf_to_ltf(prices, df_1w, prev_close)
    prev_high_aligned = align_htf_to_ltf(prices, df_1w, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1w, prev_low)
    
    # Calculate Camarilla levels for current week
    range_ = prev_high_aligned - prev_low_aligned
    # Camarilla H3, L3, H4, L4 (using weekly equivalents)
    h3 = prev_close_aligned + 1.1 * range_ * 1.1/4
    l3 = prev_close_aligned - 1.1 * range_ * 1.1/4
    h4 = prev_close_aligned + 1.1 * range_ * 1.1/2
    l4 = prev_close_aligned - 1.1 * range_ * 1.1/2
    
    # Volume confirmation - 24-period average volume (2 days of 12h data)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or np.isnan(h4[i]) or np.isnan(l4[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: break above H3 with trend alignment and volume spike
            if (close[i] > h3[i] and 
                close[i] > ema50_1w_aligned[i] and
                vol_ratio[i] > 1.8):
                signals[i] = 0.25
                position = 1
            # Short entry: break below L3 with trend alignment and volume spike
            elif (close[i] < l3[i] and 
                  close[i] < ema50_1w_aligned[i] and
                  vol_ratio[i] > 1.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below L4 (mean reversion) OR trend fails
            if close[i] < l4[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above H4 (mean reversion) OR trend fails
            if close[i] > h4[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals