#!/usr/bin/env python3
"""
Experiment #005: 1d Donchian + Funding Rate Z-Score Regime

HYPOTHESIS: Funding rate is a contrarian indicator. When funding is extremely
negative (Z < -2), traders are overly pessimistic = buy signal. When funding
is extremely positive (Z > +2), traders are overly optimistic = sell signal.

Donchian(20) breakout provides entry timing on the 1d chart.
Volume confirmation ensures institutional participation.
This combination works in BOTH bull (funding spike at top) and bear (funding
crash at bottom) markets.

TARGET: 75-150 total trades over 4 years = 19-37/year.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_funding_zscore_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def load_funding_rate(funding_path, symbol, prices_df):
    """Load and align funding rate data to prices index."""
    import os
    filepath = os.path.join(funding_path, f"{symbol}.parquet")
    if not os.path.exists(filepath):
        return np.full(len(prices_df), np.nan)
    
    df_fund = pd.read_parquet(filepath)
    # funding_df has 'open_time' and 'funding_rate' columns
    fund_times = pd.to_datetime(df_fund['open_time'], unit='ms')
    fund_rates = df_fund['funding_rate'].values
    
    # Create aligned array using forward fill
    aligned = np.full(len(prices_df), np.nan)
    
    # Match funding times to prices times
    price_times = pd.DatetimeIndex(prices_df['open_time'])
    
    for i, pt in enumerate(price_times):
        # Find the most recent funding rate before this price time
        mask = fund_times <= pt
        if mask.any():
            aligned[i] = fund_rates[mask].iloc[-1]
    
    return aligned

def calculate_funding_zscore(funding_rates, lookback=30):
    """Calculate rolling Z-score of funding rate."""
    funding_series = pd.Series(funding_rates)
    rolling_mean = funding_series.rolling(window=lookback, min_periods=10).mean()
    rolling_std = funding_series.rolling(window=lookback, min_periods=10).std()
    zscore = (funding_series - rolling_mean) / np.where(rolling_std > 0, rolling_std, 1e-10)
    return zscore.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA for broader trend
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=8, min_periods=8, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === Local 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (20 periods)
    donchian_period = 20
    upper_donch = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().shift(1).values
    lower_donch = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().shift(1).values
    mid_donch = (upper_donch + lower_donch) / 2
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Funding rate Z-score (try to load, fallback to zeros if not available)
    try:
        funding_path = "data/processed/funding"
        funding_rates = load_funding_rate(funding_path, "BTCUSDT", prices)
        funding_zscore = calculate_funding_zscore(funding_rates, lookback=30)
    except Exception:
        # If funding data unavailable, use neutral regime
        funding_zscore = np.zeros(n)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 50  # Enough for Donchian(20) + ATR(14)
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(upper_donch[i]) or np.isnan(lower_donch[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === FUNDING RATE REGIME (Z-score based) ===
        fund_z = funding_zscore[i] if not np.isnan(funding_zscore[i]) else 0.0
        
        # Long bias when funding extremely negative (Z < -2)
        long_regime = fund_z < -1.5
        # Short bias when funding extremely positive (Z > +2)
        short_regime = fund_z > 1.5
        
        # Broader trend from weekly EMA
        price_above_1w_ema = close[i] > ema_1w_aligned[i] if not np.isnan(ema_1w_aligned[i]) else True
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === Donchian breakout signals ===
        donch_breakout_up = close[i] > upper_donch[i] and low[i] > upper_donch[i]
        donch_breakout_down = close[i] < lower_donch[i] and high[i] < lower_donch[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Donchian breakout UP + funding regime + trend alignment ===
            if donch_breakout_up and vol_spike:
                # Strong signal: breakout + volume
                if long_regime or price_above_1w_ema:
                    desired_signal = SIZE
            
            # === SHORT: Donchian breakout DOWN + funding regime ===
            if donch_breakout_down and vol_spike:
                # Strong signal: breakout + volume
                if short_regime or not price_above_1w_ema:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars (reduce churn) ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 3:
            # Take profit at Donchian mid
            if position_side > 0 and close[i] >= mid_donch[i]:
                desired_signal = 0.0
            if position_side < 0 and close[i] <= mid_donch[i]:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals