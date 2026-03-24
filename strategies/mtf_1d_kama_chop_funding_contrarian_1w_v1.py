#!/usr/bin/env python3
"""
Experiment #673: 1d Primary + 1w HTF — KAMA + Choppiness + Funding Contrarian

Hypothesis: Daily timeframe with weekly HTF filter provides optimal signal quality
for crypto perpetuals. Key innovations vs failed strategies:
1. KAMA (Kaufman Adaptive MA) — adapts to volatility, better than HMA/EMA in chop
2. Choppiness Index regime — CHOP>61.8=range(mean-revert), CHOP<38.2=trend
3. Funding Rate Z-score contrarian — research shows Sharpe 0.8-1.5 for BTC/ETH
4. Weekly HMA for macro bias — prevents counter-trend in strong weekly trends
5. LOOSE entry thresholds (RSI 25/75, not 20/80) to ensure trade generation
6. Position size 0.25-0.30 with 2.5x ATR trailing stop

Why this should work where others failed:
- 446 strategies failed mostly due to 0 trades or negative Sharpe
- Funding rate contrarian is proven edge for BTC/ETH (best in bear markets)
- KAMA adapts efficiency ratio — smooth in trends, responsive in ranges
- 1d TF = ~20-40 trades/year (low fee drag, high signal quality)
- Weekly HTF prevents fighting macro trend

Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_chop_funding_contrarian_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average — adapts to market efficiency.
    Smooth in trending markets, responsive in ranging markets.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    # Efficiency Ratio (ER) — measures trend vs noise
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Smoothing Constant (SC)
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    
    # KAMA calculation
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] ** 2 * (close[i] - kama[i - 1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) — identifies ranging vs trending markets.
    CHOP > 61.8 = choppy/ranging (mean-revert)
    CHOP < 38.2 = trending (trend-follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high > lowest_low:
            atr_sum = 0.0
            for j in range(i - period + 1, i + 1):
                tr1 = high[j] - low[j]
                tr2 = np.abs(high[j] - close[j - 1]) if j > 0 else tr1
                tr3 = np.abs(low[j] - close[j - 1]) if j > 0 else tr1
                atr_sum += max(tr1, tr2, tr3)
            
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 100
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_rsi(close, period=14):
    """Relative Strength Index with proper min_periods."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_raw = 100 - (100 / (1 + rs))
        rsi[period:] = rsi_raw[period - 1:]
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands with %B for overbought/oversold."""
    n = len(close)
    bb_mid = np.full(n, np.nan)
    bb_upper = np.full(n, np.nan)
    bb_lower = np.full(n, np.nan)
    bb_pct = np.full(n, np.nan)
    
    if n < period:
        return bb_mid, bb_upper, bb_lower, bb_pct
    
    bb_mid = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    bb_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    bb_upper = bb_mid + std_mult * bb_std
    bb_lower = bb_mid - std_mult * bb_std
    
    with np.errstate(divide='ignore', invalid='ignore'):
        bb_pct = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    
    bb_pct = np.clip(bb_pct, 0, 1)
    return bb_mid, bb_upper, bb_lower, bb_pct

def calculate_hma(close, period=21):
    """Hull Moving Average — smoother than EMA, less lag."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    kama_1d = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    rsi_1d = calculate_rsi(close, period=14)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    atr_1d = calculate_atr(high, low, close, period=14)
    bb_mid, bb_upper, bb_lower, bb_pct = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Calculate and align HTF (1w) indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # RSI on weekly for macro sentiment
    rsi_1w_raw = calculate_rsi(df_1w['close'].values, period=14)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w_raw)
    
    # === FUNDING RATE CONTRARIAN SIGNAL ===
    # Load funding rate data for contrarian edge (proven for BTC/ETH)
    try:
        import os
        symbol = prices.get('symbol', 'BTCUSDT')
        if isinstance(symbol, (list, np.ndarray)):
            symbol = symbol[0] if len(symbol) > 0 else 'BTCUSDT'
        
        funding_path = f"data/processed/funding/{symbol}.parquet"
        if os.path.exists(funding_path):
            funding_df = pd.read_parquet(funding_path)
            # Align funding to prices timeline
            funding_df = funding_df.set_index('open_time').reindex(
                prices['open_time'].values, method='ffill'
            ).fillna(0)
            funding_rate = funding_df['funding_rate'].values
            
            # Z-score of funding rate (30-day rolling)
            funding_ma = pd.Series(funding_rate).rolling(window=30, min_periods=15).mean().values
            funding_std = pd.Series(funding_rate).rolling(window=30, min_periods=15).std().values
            with np.errstate(divide='ignore', invalid='ignore'):
                funding_zscore = (funding_rate - funding_ma) / (funding_std + 1e-10)
            funding_zscore = np.nan_to_num(funding_zscore, nan=0.0)
        else:
            funding_zscore = np.zeros(n)
    except Exception:
        funding_zscore = np.zeros(n)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Start after warmup period
        # Skip if indicators not ready
        if np.isnan(kama_1d[i]) or np.isnan(rsi_1d[i]):
            continue
        if np.isnan(chop_1d[i]) or np.isnan(atr_1d[i]) or atr_1d[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(rsi_1w_aligned[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop_value = chop_1d[i]
        is_range_regime = chop_value > 55  # Mean-revert in choppy markets
        is_trend_regime = chop_value < 45  # Trend-follow in trending markets
        
        # === WEEKLY MACRO BIAS (1w HMA + RSI) ===
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        weekly_rsi_neutral = 40 <= rsi_1w_aligned[i] <= 60
        
        # === DAILY TREND (KAMA) ===
        kama_bullish = close[i] > kama_1d[i]
        kama_bearish = close[i] < kama_1d[i]
        kama_slope_up = kama_1d[i] > kama_1d[i - 5] if i >= 5 else False
        kama_slope_down = kama_1d[i] < kama_1d[i - 5] if i >= 5 else False
        
        # === RSI SIGNALS (LOOSE thresholds for trade generation) ===
        rsi_oversold = rsi_1d[i] < 35
        rsi_overbought = rsi_1d[i] > 65
        rsi_extreme_oversold = rsi_1d[i] < 25
        rsi_extreme_overbought = rsi_1d[i] > 75
        
        # === BB POSITION ===
        bb_near_lower = bb_pct[i] < 0.15
        bb_near_upper = bb_pct[i] > 0.85
        
        # === FUNDING CONTRARIAN ===
        funding_extreme_long = funding_zscore[i] > 1.5  # Too bullish → short
        funding_extreme_short = funding_zscore[i] < -1.5  # Too bearish → long
        
        desired_signal = 0.0
        
        # === REGIME 1: TRENDING (CHOP < 45) — Trend Follow ===
        if is_trend_regime:
            # Long: Weekly bullish + Daily KAMA bullish + RSI not extreme
            if weekly_bullish and kama_bullish:
                if rsi_1d[i] < 65 and not rsi_overbought:
                    desired_signal = SIZE_LONG
                # Pullback entry in uptrend
                elif rsi_oversold and weekly_rsi_neutral:
                    desired_signal = SIZE_LONG
            
            # Short: Weekly bearish + Daily KAMA bearish + RSI not extreme
            elif weekly_bearish and kama_bearish:
                if rsi_1d[i] > 35 and not rsi_oversold:
                    desired_signal = -SIZE_SHORT
                # Rally entry in downtrend
                elif rsi_overbought and weekly_rsi_neutral:
                    desired_signal = -SIZE_SHORT
        
        # === REGIME 2: RANGING (CHOP > 55) — Mean Reversion ===
        elif is_range_regime:
            # Long: RSI oversold + BB near lower + Funding extreme short
            if rsi_extreme_oversold or (rsi_oversold and bb_near_lower):
                if funding_extreme_short or weekly_bullish:
                    desired_signal = SIZE_LONG
            
            # Short: RSI overbought + BB near upper + Funding extreme long
            if rsi_extreme_overbought or (rsi_overbought and bb_near_upper):
                if funding_extreme_long or weekly_bearish:
                    desired_signal = -SIZE_SHORT
        
        # === REGIME 3: TRANSITION (45 <= CHOP <= 55) — Mixed ===
        else:
            # Use KAMA direction with RSI filter
            if kama_bullish and rsi_1d[i] < 60 and weekly_bullish:
                desired_signal = SIZE_LONG
            elif kama_bearish and rsi_1d[i] > 40 and weekly_bearish:
                desired_signal = -SIZE_SHORT
        
        # === FUNDING CONTRARIAN OVERRIDE ===
        # Strong funding signal can override regime
        if funding_extreme_short and rsi_oversold and desired_signal <= 0:
            desired_signal = SIZE_LONG * 0.5  # Smaller size for contrarian
        elif funding_extreme_long and rsi_overbought and desired_signal >= 0:
            desired_signal = -SIZE_SHORT * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if KAMA still bullish AND RSI not extremely overbought
                if kama_bullish and rsi_1d[i] < 75:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if KAMA still bearish AND RSI not extremely oversold
                if kama_bearish and rsi_1d[i] > 25:
                    desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            # If same side, update trailing stop levels
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