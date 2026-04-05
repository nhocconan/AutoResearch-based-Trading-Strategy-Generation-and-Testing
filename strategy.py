# 0.027 (41 tr/sym) ← too_few_trades(41tr)
  #7714 [   keep] exp_7714_4h_donchian20_1d_ema_vol_v1 | Sharpe=0.027 (41 tr/sym)
  #7716 [   keep] exp_7716_4h_donchian20_1d_ema_vol_v1 | Sharpe=0.027 (41 tr/sym)

⚠️ REJECTED: 41 total trades over 4 years = 10.25/year. Minimum is 12.5/year (50 total).
TARGET: 75-250 total over 4 years. You need ~40 more trades.

ISSUE: Conditions are too strict. The combination of:
1. Price above/below 1d EMA (trend filter)
2. Close beyond previous bar's Donchian band (breakout)
3. Volume > 1.5x 20-period MA
is rarely all true at once.

FIX: Relax one condition slightly while keeping risk controls. Options:
- Reduce volume threshold from 1.5 to 1.3
- Allow breakout on either high or low (not just close)
- Use 15-period Donchian instead of 20

Choose ONE relaxation to add ~40 trades while keeping edge.

# 0.027 (41 tr/sym) ← too_few_trades(41tr)
  #7714 [   keep] exp_7714_4h_donchian20_1d_ema_vol_v1 | Sharpe=0.027 (41 tr/sym)
  #7716 [   keep] exp_7716_4h_donchian20_1d_ema_vol_v1 | Sharpe=0.027 (41 tr/sym)

⚠️ REJECTED: 41 total trades over 4 years = 10.25/year. Minimum is 12.5/year (50 total).
TARGET: 75-250 total over 4 years. You need ~40 more trades.

ISSUE: Conditions are too strict. The combination of:
1. Price above/below 1d EMA (trend filter)
2. Close beyond previous bar's Donchian band (breakout)
3. Volume > 1.5x 20-period MA
is rarely all true at once.

FIX: Relax one condition slightly while keeping risk controls. Options:
- Reduce volume threshold from 1.5 to 1.3
- Allow breakout on either high or low (not just close)
- Use 15-period Donchian instead of 20

Choose ONE relaxation to add ~40 trades while keeping edge.

</think>
#!/usr/bin/env python3
"""
Experiment #7713: 4-hour Donchian(18) breakout with 1-day EMA trend filter and volume confirmation.
Relaxed Donchian period from 20 to 18 to increase trade frequency while maintaining edge.
Hypothesis: Price breaking beyond 18-period high/low on 4h with volume confirmation and aligned 1d trend
captures sustained moves while avoiding whipsaw. Works in bull markets (long breakouts above EMA) and bear
markets (short breakdowns below EMA). Targets 75-200 trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7713_4h_donchian18_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 18  # Reduced from 20 to increase trade frequency
EMA_TREND = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_TREND, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss or target
        if position == 1:  # long position
            if close[i] <= stop_price or close[i] >= target_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or close[i] <= target_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market regime
        bull_regime = close[i] > ema_1d_aligned[i]   # price above 1d EMA
        bear_regime = close[i] < ema_1d_aligned[i]   # price below 1d EMA
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions - require close beyond Donchian bands to avoid wicks
        upper_breakout = (close[i] > highest_high[i-1]) and (i-1 >= 0) and not np.isnan(highest_high[i-1])
        lower_breakout = (close[i] < lowest_low[i-1]) and (i-1 >= 0) and not np.isnan(lowest_low[i-1])
        
        # Entry conditions
        long_entry = bull_regime and upper_breakout and volume_confirmed
        short_entry = bear_regime and lower_breakout and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (ATR_TARGET_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price - (ATR_TARGET_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals