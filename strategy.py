# 1d DMI (ADX) with volume confirmation
# Hypothesis: DMI identifies strong trends (ADX>25) and direction (+DI/-DI), while volume confirms institutional participation.
# Works in bull via +DI crossovers, in bear via -DI crossovers. ADX filters out weak/choppy markets.
# Target: 15-30 trades/year to minimize fee drag.
name = "1d_dmi_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly 50-period EMA for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate DMI (ADX) components
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values
    tr_ma = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / tr_ma
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / tr_ma
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: daily volume > weekly average volume
    vol_1w = df_1w['volume'].values
    vol_avg_1w = pd.Series(vol_1w).rolling(window=4, min_periods=4).mean().values  # ~monthly average
    vol_avg_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_1w)
    vol_confirm = volume > vol_avg_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: ADX weakens or trend reverses
            if adx[i] < 20 or minus_di[i] > plus_di[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: ADX weakens or trend reverses
            if adx[i] < 20 or plus_di[i] > minus_di[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: +DI crosses above -DI, ADX>25, uptrend, volume confirmation
            if (plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1] and 
                adx[i] > 25 and uptrend and vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: -DI crosses above +DI, ADX>25, downtrend, volume confirmation
            elif (minus_di[i] > plus_di[i] and minus_di[i-1] <= plus_di[i-1] and 
                  adx[i] > 25 and downtrend and vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals