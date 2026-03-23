#!/usr/bin/env python3
"""
Experiment #1304: 4h Primary + 12h/1d HTF — Funding Rate + Fisher Transform Reversal

Hypothesis: Research notes indicate funding rate mean reversion is the BEST edge for BTC/ETH
(Sharpe 0.8-1.5 through 2022 crash). Combined with Ehlers Fisher Transform for reversal
timing and 12h HMA for macro trend filter, this should work in both bull and bear markets.

Key components:
1. FUNDING RATE Z-SCORE: z > +2.0 → short extreme (crowded longs), z < -2.0 → long
2. EHLERS FISHER TRANSFORM: period=9, crosses -1.5 (long) / +1.5 (short) for timing
3. 12h HMA: Macro trend filter (only long if price > 12h HMA, short if below)
4. CHOPPINESS INDEX: Regime filter (mean-revert in chop > 55, trend-follow < 45)
5. ATR trailing stop: 2.5x ATR to protect capital

Target: Sharpe > 0.612, trades >= 40 train, >= 6 test, DD > -40%
Timeframe: 4h
Size: 0.25-0.30 discrete levels
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_funding_fisher_regime_12h_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_fisher_transform(high, low, close, period=9):
    """Ehlers Fisher Transform - catches reversals in bear rallies
    Long when Fisher crosses above -1.5 from below
    Short when Fisher crosses below +1.5 from above
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_prev
    
    for i in range(period, n):
        hl2 = (high[i] + low[i]) / 2.0
        
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        if hh > ll:
            x = (hl2 - ll) / (hh - ll)
            x = np.clip(x, 0.001, 0.999)
            
            fisher_val = 0.5 * np.log((1.0 + x) / (1.0 - x))
            
            if i > period:
                fisher[i] = 0.67 * fisher_val + 0.33 * fisher[i-1]
                fisher_prev[i] = fisher[i-1]
            else:
                fisher[i] = fisher_val
                fisher_prev[i] = fisher_val
    
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - regime detection
    CHOP > 61.8 = ranging (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        tr_sum = np.sum(tr[i-period+1:i+1])
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        if hh > ll and tr_sum > 0:
            chop[i] = 100.0 * np.log10(tr_sum / (hh - ll)) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[:period] = np.nan
    
    return rsi

def load_funding_data(symbol):
    """Load funding rate data from parquet file"""
    try:
        # Extract symbol from prices (assume it's in index or we derive from context)
        # Funding data path: data/processed/funding/{symbol}.parquet
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        funding_path = os.path.join(base_dir, '..', 'data', 'processed', 'funding', f'{symbol}.parquet')
        
        if os.path.exists(funding_path):
            df_funding = pd.read_parquet(funding_path)
            return df_funding
        else:
            # Try alternate path
            funding_path = f'data/processed/funding/{symbol}.parquet'
            if os.path.exists(funding_path):
                df_funding = pd.read_parquet(funding_path)
                return df_funding
    except Exception:
        pass
    
    return None

def calculate_funding_zscore(funding_rates, window=30):
    """Calculate z-score of funding rates over rolling window"""
    n = len(funding_rates)
    zscore = np.full(n, np.nan)
    
    if n < window:
        return zscore
    
    funding_series = pd.Series(funding_rates)
    
    for i in range(window - 1, n):
        window_data = funding_series.iloc[i-window+1:i+1]
        mean = window_data.mean()
        std = window_data.std()
        
        if std > 1e-10:
            zscore[i] = (funding_rates[i] - mean) / std
    
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for macro trend filter
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Try to load funding rate data
    funding_zscore = None
    try:
        # Derive symbol from prices DataFrame
        symbol = 'BTCUSDT'  # Default, will be overridden by actual symbol
        if hasattr(prices, 'symbol'):
            symbol = prices.symbol
        elif 'symbol' in prices.columns:
            symbol = prices['symbol'].iloc[0]
        
        df_funding = load_funding_data(symbol)
        
        if df_funding is not None and len(df_funding) > 0:
            # Align funding data to prices timeframe
            # Funding is typically 8h, we need to align to 4h
            funding_rates = df_funding['funding_rate'].values if 'funding_rate' in df_funding.columns else df_funding.iloc[:, -1].values
            
            # Create aligned funding z-score array
            funding_zscore_4h = np.full(n, np.nan)
            
            # Simple alignment: map funding bars to price bars by time
            if 'open_time' in df_funding.columns and 'open_time' in prices.columns:
                price_times = prices['open_time'].values
                funding_times = df_funding['open_time'].values
                
                funding_zscore_raw = calculate_funding_zscore(funding_rates, window=30)
                
                # Map funding z-score to each price bar
                for i in range(n):
                    pt = price_times[i]
                    # Find the most recent funding bar before this price bar
                    valid_idx = funding_times <= pt
                    if np.any(valid_idx):
                        last_funding_idx = np.where(valid_idx)[0][-1]
                        if last_funding_idx < len(funding_zscore_raw):
                            funding_zscore_4h[i] = funding_zscore_raw[last_funding_idx]
            
            funding_zscore = funding_zscore_4h
    except Exception:
        funding_zscore = None
    
    # Calculate primary (4h) indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            continue
        if np.isnan(chop[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        in_range = chop[i] > 55.0  # Ranging market (mean revert)
        in_trend = chop[i] < 45.0  # Trending market
        
        # === MACRO TREND (12h HMA) ===
        macro_bull = close[i] > hma_12h_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i]
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        fisher_long_signal = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        fisher_short_signal = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        
        # === FUNDING RATE EXTREMES (if available) ===
        funding_extreme_long = False
        funding_extreme_short = False
        
        if funding_zscore is not None and not np.isnan(funding_zscore[i]):
            funding_extreme_long = funding_zscore[i] < -2.0  # Negative funding = shorts paying, bullish contrarian
            funding_extreme_short = funding_zscore[i] > 2.0  # Positive funding = longs paying, bearish contrarian
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # MEAN REVERSION REGIME (choppy market)
        if in_range:
            # Long: Fisher long cross + RSI oversold + macro not strongly bear
            if fisher_long_signal and rsi[i] < 40.0 and not macro_bear:
                desired_signal = BASE_SIZE
            # Short: Fisher short cross + RSI overbought + macro not strongly bull
            elif fisher_short_signal and rsi[i] > 60.0 and not macro_bull:
                desired_signal = -BASE_SIZE
            
            # Funding rate contrarian override (stronger signal)
            if funding_extreme_long and fisher_long_signal:
                desired_signal = BASE_SIZE
            elif funding_extreme_short and fisher_short_signal:
                desired_signal = -BASE_SIZE
        
        # TRENDING REGIME
        elif in_trend:
            # Long: Fisher long + macro bull
            if fisher_long_signal and macro_bull:
                desired_signal = BASE_SIZE
            # Short: Fisher short + macro bear
            elif fisher_short_signal and macro_bear:
                desired_signal = -BASE_SIZE
            
            # Funding extreme can override trend if very extreme
            if funding_extreme_long and rsi[i] < 35.0:
                desired_signal = BASE_SIZE
            elif funding_extreme_short and rsi[i] > 65.0:
                desired_signal = -BASE_SIZE
        
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
        if desired_signal > 0.1:
            final_signal = BASE_SIZE
        elif desired_signal < -0.1:
            final_signal = -BASE_SIZE
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