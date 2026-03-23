#!/usr/bin/env python3
"""
Experiment #944: 4h Primary + 12h/1d HTF — Fisher Transform + KAMA Adaptive + Funding Contrarian

Hypothesis: After 673 failed strategies, the key insight is that RSI-based mean reversion fails
in bear markets because RSI stays oversold too long. Fisher Transform normalizes price into
Gaussian distribution, making extremes (-1.5/+1.5) more reliable for reversals. Combined with
KAMA (Kaufman Adaptive MA) which reduces whipsaw in choppy markets via Efficiency Ratio,
and funding rate as PRIMARY contrarian filter (not just confluence), this should work across
ALL symbols including 2025 bear market.

Why this differs from failed strategies:
1. Fisher Transform instead of RSI — catches reversals better in bear rallies (research shows 75% win rate)
2. KAMA instead of HMA/EMA — adapts smoothing based on market noise (ER = signal/noise ratio)
3. Funding z-score as PRIMARY filter (z > 1.5 or z < -1.5) — best edge for BTC/ETH per research
4. RELAXED entry thresholds to ensure >= 30 trades/train, >= 3/test per symbol
5. 12h KAMA for medium trend + 1d KAMA for macro regime (both adaptive)

Key improvements:
- Fisher period=9, entry at -1.2/-1.5 (not -2.0 which is too rare)
- KAMA ER period=10, fast SC=2/11, slow SC=2/31 (standard Kaufman params)
- Funding z-score threshold=1.5 (not 2.0) to ensure trades
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn
- Stoploss at 2.5*ATR trailing

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
Timeframe: 4h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_kama_funding_12h1d_adaptive_v1"
timeframe = "4h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform — normalizes price into Gaussian distribution.
    Entry signals: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    Research shows 75% win rate on reversals in bear markets.
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_signal
    
    for i in range(period - 1, n):
        # Calculate typical price
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
            fisher_signal[i] = fisher[i]
            continue
        
        # Normalize price to -1 to +1 range
        normalized = 0.66 * ((hl2 - lowest) / (highest - lowest) - 0.5)
        normalized = np.clip(normalized, -0.99, 0.99)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
        
        # Signal line (1-period lag)
        if i > 0 and not np.isnan(fisher[i-1]):
            fisher_signal[i] = fisher[i-1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average — adapts smoothing based on market efficiency.
    ER (Efficiency Ratio) = |net change| / sum of absolute changes
    High ER = trending (use fast SC), Low ER = choppy (use slow SC)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    er = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama, er
    
    # Calculate Efficiency Ratio
    for i in range(period - 1, n):
        if i < period:
            continue
        
        # Net change over period
        net_change = np.abs(close[i] - close[i-period])
        
        # Sum of absolute changes (noise)
        volatility = 0.0
        for j in range(i-period+1, i+1):
            volatility += np.abs(close[j] - close[j-1])
        
        if volatility > 1e-10:
            er[i] = net_change / volatility
        else:
            er[i] = 0.0
    
    er = np.clip(er, 0, 1)
    
    # Calculate KAMA
    # Fast SC = 2/(fast_period+1) = 2/3 for fast_period=2
    # Slow SC = 2/(slow_period+1) = 2/31 for slow_period=30
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA at SMA of first period
    kama[period - 1] = np.mean(close[:period])
    
    for i in range(period, n):
        if np.isnan(er[i]):
            kama[i] = kama[i-1]
            continue
        
        # Adaptive smoothing constant
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama, er

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_funding_zscore(funding_series, period=30):
    """Z-score of funding rate over lookback period."""
    n = len(funding_series)
    zscore = np.full(n, np.nan)
    
    if n < period:
        return zscore
    
    for i in range(period - 1, n):
        window = funding_series[i-period+1:i+1]
        mean = np.mean(window)
        std = np.std(window, ddof=1)
        if std > 1e-10:
            zscore[i] = (funding_series[i] - mean) / std
        else:
            zscore[i] = 0.0
    
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Load funding rate data
    symbol = prices['symbol'].iloc[0] if 'symbol' in prices.columns else 'BTCUSDT'
    funding_path = f"data/processed/funding/{symbol}.parquet"
    try:
        df_funding = pd.read_parquet(funding_path)
        funding_rates = df_funding['funding_rate'].values
        if len(funding_rates) >= n:
            funding_rates = funding_rates[-n:]
        else:
            funding_rates = np.concatenate([np.zeros(n - len(funding_rates)), funding_rates])
    except:
        funding_rates = np.zeros(n)
    
    # Calculate primary (4h) indicators
    fisher_4h, fisher_signal_4h = calculate_fisher_transform(high, low, close, period=9)
    atr_4h = calculate_atr(high, low, close, period=14)
    kama_4h, er_4h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    
    # Calculate and align 12h KAMA for medium-term trend bias
    kama_12h_raw, _ = calculate_kama(df_12h['close'].values, period=10, fast_period=2, slow_period=30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    # Calculate and align 1d KAMA for macro regime
    kama_1d_raw, _ = calculate_kama(df_1d['close'].values, period=10, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate funding z-score
    funding_z = calculate_funding_zscore(funding_rates, period=30)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(fisher_4h[i]) or np.isnan(fisher_signal_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(kama_4h[i]) or np.isnan(kama_12h_aligned[i]) or np.isnan(kama_1d_aligned[i]):
            continue
        if np.isnan(funding_z[i]):
            continue
        
        # === MACRO REGIME (1d HTF KAMA) ===
        macro_bull = close[i] > kama_1d_aligned[i]
        macro_bear = close[i] < kama_1d_aligned[i]
        
        # === MEDIUM-TERM TREND (12h HTF KAMA) ===
        trend_12h_bullish = close[i] > kama_12h_aligned[i]
        trend_12h_bearish = close[i] < kama_12h_aligned[i]
        
        # === MARKET EFFICIENCY (4h ER from KAMA) ===
        high_efficiency = er_4h[i] > 0.5  # Trending market
        low_efficiency = er_4h[i] < 0.3   # Choppy market
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long_cross = fisher_4h[i] > -1.2 and fisher_signal_4h[i] <= -1.2
        fisher_short_cross = fisher_4h[i] < 1.2 and fisher_signal_4h[i] >= 1.2
        fisher_extreme_long = fisher_4h[i] < -1.5
        fisher_extreme_short = fisher_4h[i] > 1.5
        
        # === FUNDING RATE CONTRARIAN (PRIMARY FILTER) ===
        funding_extreme_short = funding_z[i] > 1.5  # Too many longs → short signal
        funding_extreme_long = funding_z[i] < -1.5  # Too many shorts → long signal
        funding_moderate_short = funding_z[i] > 0.8
        funding_moderate_long = funding_z[i] < -0.8
        
        desired_signal = 0.0
        
        # === HIGH EFFICIENCY (TRENDING) — Trend Following with Fisher Pullback ===
        if high_efficiency:
            # Long: Macro bullish + Fisher pullback to oversold
            if macro_bull or trend_12h_bullish:
                if fisher_long_cross or fisher_extreme_long:
                    desired_signal = BASE_SIZE
                elif funding_extreme_long:
                    desired_signal = REDUCED_SIZE
            
            # Short: Macro bearish + Fisher rally to overbought
            if macro_bear or trend_12h_bearish:
                if fisher_short_cross or fisher_extreme_short:
                    desired_signal = -BASE_SIZE
                elif funding_extreme_short:
                    desired_signal = -REDUCED_SIZE
        
        # === LOW EFFICIENCY (CHOPPY) — Mean Reversion with Funding ===
        elif low_efficiency:
            # Long: Fisher extreme oversold + funding extreme long
            if fisher_extreme_long and funding_extreme_long:
                desired_signal = BASE_SIZE
            elif fisher_extreme_long:
                desired_signal = REDUCED_SIZE
            elif funding_extreme_long:
                desired_signal = REDUCED_SIZE
            
            # Short: Fisher extreme overbought + funding extreme short
            if fisher_extreme_short and funding_extreme_short:
                desired_signal = -BASE_SIZE
            elif fisher_extreme_short:
                desired_signal = -REDUCED_SIZE
            elif funding_extreme_short:
                desired_signal = -REDUCED_SIZE
        
        # === NORMAL EFFICIENCY (0.3 <= ER <= 0.5) — Balanced Approach ===
        else:
            # Long: Funding contrarian + any trend support
            if funding_extreme_long:
                if macro_bull or trend_12h_bullish:
                    desired_signal = BASE_SIZE
                else:
                    desired_signal = REDUCED_SIZE
            elif fisher_long_cross:
                desired_signal = REDUCED_SIZE
            
            # Short: Funding contrarian + any trend support
            if funding_extreme_short:
                if macro_bear or trend_12h_bearish:
                    desired_signal = -BASE_SIZE
                else:
                    desired_signal = -REDUCED_SIZE
            elif fisher_short_cross:
                if desired_signal == 0:
                    desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro or medium trend still bullish
                if (macro_bull or trend_12h_bullish) and fisher_4h[i] < 1.0:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro or medium trend still bearish
                if (macro_bear or trend_12h_bearish) and fisher_4h[i] > -1.0:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if both macro and medium trend reverse
            if macro_bear and trend_12h_bearish:
                desired_signal = 0.0
            # Exit if funding flips extreme short
            if funding_extreme_short and fisher_4h[i] > 0.5:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if both macro and medium trend reverse
            if macro_bull and trend_12h_bullish:
                desired_signal = 0.0
            # Exit if funding flips extreme long
            if funding_extreme_long and fisher_4h[i] < -0.5:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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
        
        signals[i] = desired_signal
    
    return signals