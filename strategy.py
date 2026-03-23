#!/usr/bin/env python3
"""
Experiment #818: 30m Primary + 4h/1d HTF — Funding Z-Score + Choppiness + Session

Hypothesis: After 558 failed strategies, the key insight is that funding rate mean
reversion has proven Sharpe 0.8-1.5 on BTC/ETH through 2022 crash. Combined with:
1. Funding rate z-score(30d) extremes capture sentiment reversals
2. Choppiness Index filters regime (range vs trend)
3. 4h/1d HMA for trend direction bias
4. Session filter (8-20 UTC) for high liquidity entries
5. Volume confirmation to avoid fake breakouts

This targets bear/range markets where pure trend following fails.
30m primary with 4h/1d HTF direction = HTF trade frequency with 30m timing precision.

Key design:
- Funding z-score < -2.0 → long bias, > +2.0 → short bias
- CHOP > 55 = range (mean revert), CHOP < 40 = trend (follow)
- 4h HMA(21) + 1d HMA(21) for trend confirmation
- Session: only trade 8-20 UTC (high liquidity, less manipulation)
- Volume > 0.8x 20-bar average
- Discrete signals: 0.0, ±0.20, ±0.30
- ATR(14) trailing stop at 2.5x

Target: 40-80 trades/year, Sharpe > 0.612, ALL symbols positive
Timeframe: 30m (use 4h/1d for direction, 30m for entry timing)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_funding_zscore_chop_session_4h1d_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index — CHOP > 55 = range, CHOP < 40 = trend."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

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

def calculate_zscore(series, period=30):
    """Z-score of a series over rolling window."""
    n = len(series)
    zscore = np.full(n, np.nan)
    
    for i in range(period, n):
        window = series[i-period+1:i+1]
        mean = np.mean(window)
        std = np.std(window)
        if std > 1e-10:
            zscore[i] = (series[i] - mean) / std
        else:
            zscore[i] = 0.0
    
    return zscore

def load_funding_data(symbol):
    """Load funding rate data from processed parquet."""
    import os
    # Map symbol to filename
    symbol_map = {
        'BTCUSDT': 'BTCUSDT',
        'ETHUSDT': 'ETHUSDT',
        'SOLUSDT': 'SOLUSDT'
    }
    base_symbol = symbol_map.get(symbol, symbol.replace('USDT', ''))
    funding_path = f"data/processed/funding/{base_symbol}.parquet"
    
    try:
        if os.path.exists(funding_path):
            df_funding = pd.read_parquet(funding_path)
            return df_funding
    except:
        pass
    return None

def get_funding_zscore(prices, symbol, period=30):
    """Get funding rate z-score aligned to prices."""
    df_funding = load_funding_data(symbol)
    
    if df_funding is None or len(df_funding) == 0:
        # Fallback: use RSI as proxy for sentiment
        close = prices['close'].values
        rsi = calculate_rsi(close, 14)
        # Convert RSI to pseudo z-score (RSI 50 = 0, RSI 20 = -2, RSI 80 = +2)
        zscore = (rsi - 50) / 15
        return zscore
    
    # Extract funding rates
    funding_rates = df_funding['funding_rate'].values if 'funding_rate' in df_funding.columns else df_funding.iloc[:, -1].values
    
    # Calculate z-score on funding
    funding_zscore = calculate_zscore(funding_rates, period)
    
    # Align to prices (funding is 8h, prices are 30m = 16 funding bars per day)
    # Use simple interpolation/forward-fill approach
    n_prices = len(prices)
    n_funding = len(funding_zscore)
    
    if n_funding == 0:
        return np.zeros(n_prices)
    
    # Ratio of timeframes (30m vs 8h funding)
    ratio = n_prices / n_funding
    
    aligned_zscore = np.zeros(n_prices)
    for i in range(n_prices):
        funding_idx = min(int(i / ratio), n_funding - 1)
        aligned_zscore[i] = funding_zscore[funding_idx] if not np.isnan(funding_zscore[funding_idx]) else 0.0
    
    return aligned_zscore

def get_hour_from_open_time(open_times):
    """Extract UTC hour from open_time (milliseconds)."""
    # open_time is in milliseconds since epoch
    hours = (open_times // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_times = prices["open_time"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Extract symbol from prices metadata if available
    symbol = prices.get('symbol', ['BTCUSDT'])[0] if hasattr(prices, 'get') else 'BTCUSDT'
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (30m) indicators
    rsi_30m = calculate_rsi(close, 14)
    chop_30m = calculate_choppiness(high, low, close, 14)
    atr_30m = calculate_atr(high, low, close, 14)
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align 4h HMA for intermediate trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for long-term trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Get funding z-score (or RSI proxy if funding unavailable)
    funding_zscore = get_funding_zscore(prices, symbol, period=30)
    
    # Get UTC hour for session filter
    utc_hours = get_hour_from_open_time(open_times)
    
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
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(rsi_30m[i]) or np.isnan(chop_30m[i]) or np.isnan(atr_30m[i]):
            continue
        if atr_30m[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_200[i]) or np.isnan(vol_avg_20[i]):
            continue
        if np.isnan(funding_zscore[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= utc_hours[i] <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg_20[i]
        
        # === LONG-TERM TREND BIAS (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === SECULAR TREND FILTER (SMA200) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === REGIME DETECTION (30m Choppiness Index) ===
        ranging_regime = chop_30m[i] > 55
        trending_regime = chop_30m[i] < 40
        
        # === FUNDING Z-SCORE SIGNALS ===
        funding_extreme_long = funding_zscore[i] < -2.0
        funding_extreme_short = funding_zscore[i] > 2.0
        funding_moderate_long = -2.0 <= funding_zscore[i] < -1.0
        funding_moderate_short = 1.0 < funding_zscore[i] <= 2.0
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_30m[i] < 35
        rsi_overbought = rsi_30m[i] > 65
        rsi_extreme_oversold = rsi_30m[i] < 25
        rsi_extreme_overbought = rsi_30m[i] > 75
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion with Funding ===
        if ranging_regime:
            # Long: funding extreme + RSI oversold + trend alignment (3+ confluence)
            if funding_extreme_long and rsi_oversold and (above_sma200 or trend_4h_bullish):
                if in_session and volume_ok:
                    desired_signal = BASE_SIZE
            
            # Short: funding extreme + RSI overbought + trend alignment
            if funding_extreme_short and rsi_overbought and (below_sma200 or trend_4h_bearish):
                if in_session and volume_ok:
                    desired_signal = -BASE_SIZE
            
            # Moderate funding + extreme RSI (2+ confluence)
            if funding_moderate_long and rsi_extreme_oversold:
                if in_session and volume_ok:
                    desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            if funding_moderate_short and rsi_extreme_overbought:
                if in_session and volume_ok:
                    desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # === TRENDING REGIME (CHOP < 40) — Trend Following with Funding ===
        elif trending_regime:
            # Long: 1d bullish + 4h bullish + funding not extreme short
            if trend_1d_bullish and trend_4h_bullish and funding_zscore[i] < 1.5:
                if rsi_oversold and in_session and volume_ok:
                    desired_signal = BASE_SIZE
            
            # Short: 1d bearish + 4h bearish + funding not extreme long
            if trend_1d_bearish and trend_4h_bearish and funding_zscore[i] > -1.5:
                if rsi_overbought and in_session and volume_ok:
                    desired_signal = -BASE_SIZE
            
            # Pullback in trend with funding confirmation
            if trend_1d_bullish and funding_moderate_long and rsi_30m[i] < 45:
                if in_session and volume_ok:
                    desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            if trend_1d_bearish and funding_moderate_short and rsi_30m[i] > 55:
                if in_session and volume_ok:
                    desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # === NEUTRAL REGIME (40 <= CHOP <= 55) ===
        else:
            # Conservative: funding extreme + any trend + RSI confirmation
            if funding_extreme_long and (trend_4h_bullish or above_sma200) and rsi_oversold:
                if in_session and volume_ok:
                    desired_signal = REDUCED_SIZE
            
            if funding_extreme_short and (trend_4h_bearish or below_sma200) and rsi_overbought:
                if in_session and volume_ok:
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and funding not extreme short
                if (trend_4h_bullish or trend_1d_bullish) and funding_zscore[i] < 2.0:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and funding not extreme long
                if (trend_4h_bearish or trend_1d_bearish) and funding_zscore[i] > -2.0:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if funding turns extreme short or all trends reverse
            if funding_extreme_short or (trend_1d_bearish and trend_4h_bearish):
                desired_signal = 0.0
            # Exit if RSI extremely overbought in ranging regime
            if ranging_regime and rsi_extreme_overbought:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if funding turns extreme long or all trends reverse
            if funding_extreme_long or (trend_1d_bullish and trend_4h_bullish):
                desired_signal = 0.0
            # Exit if RSI extremely oversold in ranging regime
            if ranging_regime and rsi_extreme_oversold:
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
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
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