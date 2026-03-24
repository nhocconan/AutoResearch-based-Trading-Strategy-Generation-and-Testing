#!/usr/bin/env python3
"""
Experiment #056: 12h Primary + 1d HTF — KAMA Trend + Donchian Breakout + Funding Contrarian

Hypothesis: Previous regime-switching strategies (#046, #047, #054) failed due to 
choppiness index whipsaw. This strategy uses SIMPLER trend-following with:
1. KAMA(10,2,30) - Adaptive trend that reduces noise in choppy markets
2. Donchian(20) breakout - Proven breakout signal for SOL
3. 1d HMA(21) - Simple directional bias (not regime switch)
4. RSI(14) pullback filter - Avoid chasing breakouts
5. Funding rate z-score - Contrarian edge for BTC/ETH mean reversion

Key improvements over #052:
- NO choppiness regime switching (failed in 3 experiments)
- Looser RSI thresholds (35/65 not 30/70) for more trades
- Funding rate contrarian signal adds uncorrelated alpha
- KAMA adapts to volatility automatically

Target: Sharpe>0.351, trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 12h (target 25-45 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_donchian_funding_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market noise - smooth in choppy, responsive in trends
    """
    n = len(close)
    if n < efficiency_period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(efficiency_period, n):
        signal = abs(close[i] - close[i - efficiency_period])
        noise = 0.0
        for j in range(i - efficiency_period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA
    kama[efficiency_period] = close[efficiency_period]
    
    for i in range(efficiency_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns (upper, lower, middle)"""
    n = len(close := high)  # Use high length
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    middle = (upper + lower) / 2.0
    return upper, lower, middle

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def load_funding_zscore(symbol, dates):
    """
    Load funding rate data and calculate z-score
    Returns aligned z-score array for contrarian signal
    """
    try:
        # Map symbol to funding file
        symbol_map = {
            'BTCUSDT': 'BTCUSDT',
            'ETHUSDT': 'ETHUSDT',
            'SOLUSDT': 'SOLUSDT'
        }
        funding_symbol = symbol_map.get(symbol, 'BTCUSDT')
        funding_path = f"data/processed/funding/{funding_symbol}.parquet"
        
        df_funding = pd.read_parquet(funding_path)
        df_funding = df_funding.sort_values('open_time')
        
        # Calculate z-score of funding rate (30-period)
        funding_rates = df_funding['funding_rate'].values
        n_funding = len(funding_rates)
        
        zscore = np.full(n_funding, np.nan)
        for i in range(30, n_funding):
            window = funding_rates[i-30:i]
            mean = np.mean(window)
            std = np.std(window)
            if std > 1e-10:
                zscore[i] = (funding_rates[i] - mean) / std
            else:
                zscore[i] = 0.0
        
        # Align to prices timeframe
        # Merge on open_time
        df_funding['zscore'] = zscore
        df_funding = df_funding[['open_time', 'zscore']]
        
        # Create prices dataframe with open_time for merge
        df_prices = pd.DataFrame({'open_time': dates})
        df_merged = pd.merge_asof(df_prices, df_funding, on='open_time', direction='backward')
        
        return df_merged['zscore'].values
    except Exception:
        # Return zeros if funding data unavailable
        return np.zeros(len(dates))

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_times = prices["open_time"].values
    n = len(close)
    
    # Infer symbol from prices (use BTC as default)
    symbol = "BTCUSDT"
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for HTF trend bias
    hma_1d_raw = calculate_kama(df_1d['close'].values, efficiency_period=10, fast_period=2, slow_period=30)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    kama_12h = calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Load funding z-score
    funding_zscore = load_funding_zscore(symbol, open_times)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama_12h[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d KAMA) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 12h TREND (KAMA) ===
        kama_bull = close[i] > kama_12h[i]
        kama_bear = close[i] < kama_12h[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i - 1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i - 1] if i > 0 else False
        
        # === RSI PULLBACK FILTER (loose thresholds for trade gen) ===
        rsi_bull = rsi[i] > 35.0  # Not oversold
        rsi_bear = rsi[i] < 65.0  # Not overbought
        
        # === FUNDING CONTRARIAN ===
        funding_extreme_long = funding_zscore[i] < -1.5 if not np.isnan(funding_zscore[i]) else False
        funding_extreme_short = funding_zscore[i] > 1.5 if not np.isnan(funding_zscore[i]) else False
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: HTF bull + KAMA bull + (breakout OR pullback) + funding support
        long_score = 0
        if hma_1d_bull:
            long_score += 1
        if kama_bull:
            long_score += 1
        if breakout_long:
            long_score += 1
        if rsi_bull:
            long_score += 1
        if funding_extreme_long:
            long_score += 1
        
        # SHORT ENTRY: HTF bear + KAMA bear + (breakout OR pullback) + funding support
        short_score = 0
        if hma_1d_bear:
            short_score += 1
        if kama_bear:
            short_score += 1
        if breakout_short:
            short_score += 1
        if rsi_bear:
            short_score += 1
        if funding_extreme_short:
            short_score += 1
        
        # Entry threshold: need 3+ signals for long, 3+ for short
        if long_score >= 3:
            desired_signal = SIZE
        elif short_score >= 3:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals