# 12h_1d_1w_BollingerBandBreakout_VolumeTrend_v1
# Hypothesis: 12h Bollinger Band breakout with 1d trend filter and 1w volume confirmation
# Long: Price breaks above upper BB on 12h + 1d EMA50 uptrend + 1w volume spike
# Short: Price breaks below lower BB on 12h + 1d EMA50 downtrend + 1w volume spike
# Uses Bollinger Bands for volatility-based breakouts, EMA for trend, volume for confirmation
# Target: 15-25 trades/year (60-100 total over 4 years) to avoid fee drag
# Works in bull markets via upward breakouts, in bear via downward breakouts
name = "12h_1d_1w_BollingerBandBreakout_VolumeTrend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    # Get 1w data for volume confirmation (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1w average volume for volume confirmation
    volume_1w = df_1w['volume'].values
    avg_volume_1w = pd.Series(volume_1w).rolling(window=4, min_periods=4).mean().values  # 4-week average
    avg_volume_1w_aligned = align_htf_to_ltf(prices, df_1w, avg_volume_1w, additional_delay_bars=1)
    
    # 12h Bollinger Bands (20-period, 2 std dev)
    bb_period = 20
    bb_std = 2
    sma_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_bb + (bb_std * std_bb)
    lower_bb = sma_bb - (bb_std * std_bb)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 50)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or \
           np.isnan(avg_volume_1w_aligned[i]) or \
           np.isnan(sma_bb[i]) or np.isnan(std_bb[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volume confirmation: current 12h volume > 1.5x 4-week average 1w volume
        volume_spike = volume[i] > 1.5 * avg_volume_1w_aligned[i]
        
        if position == 0:
            # Long: Price breaks above upper BB + 1d uptrend + volume spike
            if price > upper_bb[i] and price > ema50_1d_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower BB + 1d downtrend + volume spike
            elif price < lower_bb[i] and price < ema50_1d_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price breaks below middle BB or 1d trend reverses
            if price < sma_bb[i] or price < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price breaks above middle BB or 1d trend reverses
            if price > sma_bb[i] or price > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals