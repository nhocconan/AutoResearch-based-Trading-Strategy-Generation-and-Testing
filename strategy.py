#!/usr/bin/env python3
"""
Experiment #923: 1d Primary + 1w HTF — KAMA Adaptive Trend + ADX Regime + RSI

Hypothesis: After 654 failed strategies, daily timeframe with adaptive indicators
should work better than fixed EMAs. Key insights:

1. KAMA (Kaufman Adaptive MA) adapts to volatility — works in both trend/range
2. ADX(14) regime: >25=trend (breakout), <20=range (mean revert)
3. 1w HMA(21) for macro bias — only trade with weekly trend
4. Relaxed RSI thresholds (30/70 not 20/80) to ensure trades on ALL symbols
5. Donchian(20) breakout in trending regime
6. Funding rate z-score contrarian overlay for BTC/ETH edge

Why 1d should work:
- Target 20-50 trades/year = 80-200 trades over 4-year train
- Less noise than 4h/12h, fewer whipsaws
- 1w HTF provides strong macro filter
- KAMA adapts better than HMA/EMA in crypto volatility

Critical fixes from failures:
- RELAXED entry thresholds (RSI 30/70, ADX 20/25 hysteresis)
- Simplified regime logic (2 states not 3-5)
- Funding rate as soft filter not hard requirement
- Discrete signal sizes (0.0, ±0.25, ±0.30)
- ATR 2.5x trailing stop mandatory

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_adx_regime_rsi_1w_hma_funding_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency (trend vs noise)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    # Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # KAMA calculation
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

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

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    ADX > 25 = trending, ADX < 20 = ranging
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        elif down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smoothed DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * plus_dm_s / (tr_s + 1e-10)
        minus_di = 100 * minus_dm_s / (tr_s + 1e-10)
    
    # DX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # ADX (smoothed DX)
    adx_raw = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx = adx_raw
    
    return adx

def calculate_donchian(high, low, period=20):
    """Donchian Channels — highest high and lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

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
    """Z-score of a series."""
    n = len(series)
    zscore = np.full(n, np.nan)
    
    for i in range(period, n):
        window = series[i-period:i]
        mean = np.mean(window)
        std = np.std(window)
        if std > 0:
            zscore[i] = (series[i] - mean) / std
        else:
            zscore[i] = 0
    
    return zscore

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
    adx_1d = calculate_adx(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_1d = calculate_atr(high, low, close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align 1w HMA for macro regime (bull/bear market)
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Try to load funding rate data (optional, for BTC/ETH edge)
    funding_zscore = None
    try:
        import os
        symbol = prices.get('symbol', 'BTCUSDT')
        if isinstance(symbol, pd.Series):
            symbol = symbol.iloc[0]
        funding_path = f"data/processed/funding/{symbol}.parquet"
        if os.path.exists(funding_path):
            funding_df = pd.read_parquet(funding_path)
            if 'funding_rate' in funding_df.columns:
                funding_rates = funding_df['funding_rate'].values
                # Align funding to prices length
                if len(funding_rates) >= n:
                    funding_zscore = calculate_zscore(funding_rates[:n], period=30)
    except:
        funding_zscore = None
    
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
        if np.isnan(kama_1d[i]) or np.isnan(rsi_1d[i]) or np.isnan(adx_1d[i]):
            continue
        if np.isnan(atr_1d[i]) or atr_1d[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === MACRO REGIME (1w HTF HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === ADX REGIME (1d) ===
        # Hysteresis: enter trend at 25, exit at 20
        trending_regime = adx_1d[i] > 25
        ranging_regime = adx_1d[i] < 20
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama_1d[i]
        kama_bearish = close[i] < kama_1d[i]
        
        # === SMA TREND FILTER ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI SIGNALS (Relaxed: 30/70) ===
        rsi_oversold = rsi_1d[i] < 30
        rsi_overbought = rsi_1d[i] > 70
        rsi_extreme_oversold = rsi_1d[i] < 25
        rsi_extreme_overbought = rsi_1d[i] > 75
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === FUNDING RATE CONTRARIAN (BTC/ETH edge) ===
        funding_bullish = False
        funding_bearish = False
        if funding_zscore is not None and i < len(funding_zscore) and not np.isnan(funding_zscore[i]):
            funding_bullish = funding_zscore[i] < -1.5  # Extreme negative funding → long
            funding_bearish = funding_zscore[i] > 1.5   # Extreme positive funding → short
        
        desired_signal = 0.0
        
        # === TRENDING REGIME (ADX > 25) — Breakout with trend ===
        if trending_regime:
            # Long: Macro bull + KAMA bull + Donchian breakout
            if macro_bull and kama_bullish and donchian_breakout_long:
                desired_signal = BASE_SIZE
            # Long: Macro bull + KAMA bull + RSI pullback
            elif macro_bull and kama_bullish and rsi_oversold:
                desired_signal = REDUCED_SIZE
            # Long: Strong trend alignment (3+ filters)
            elif macro_bull and kama_bullish and above_sma50 and above_sma200:
                desired_signal = REDUCED_SIZE
            
            # Short: Macro bear + KAMA bear + Donchian breakdown
            if macro_bear and kama_bearish and donchian_breakout_short:
                desired_signal = -BASE_SIZE
            # Short: Macro bear + KAMA bear + RSI rally
            elif macro_bear and kama_bearish and rsi_overbought:
                desired_signal = -REDUCED_SIZE
            # Short: Strong trend alignment
            elif macro_bear and kama_bearish and below_sma50 and below_sma200:
                desired_signal = -REDUCED_SIZE
        
        # === RANGING REGIME (ADX < 20) — Mean Reversion ===
        elif ranging_regime:
            # Long: RSI oversold + macro neutral/bull
            if rsi_oversold and (macro_bull or not macro_bear):
                desired_signal = BASE_SIZE
            # Long: Extreme RSI alone (guarantees trades)
            elif rsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            # Long: Price at Donchian low + RSI low
            elif close[i] < donchian_lower[i-1] * 1.02 and rsi_1d[i] < 35:
                desired_signal = REDUCED_SIZE
            
            # Short: RSI overbought + macro neutral/bear
            if rsi_overbought and (macro_bear or not macro_bull):
                desired_signal = -BASE_SIZE
            # Short: Extreme RSI alone
            elif rsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
            # Short: Price at Donchian high + RSI high
            elif close[i] > donchian_upper[i-1] * 0.98 and rsi_1d[i] > 65:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (20 <= ADX <= 25) ===
        else:
            # Conservative: Only trade with strong confluence
            if macro_bull and kama_bullish and rsi_oversold:
                desired_signal = REDUCED_SIZE
            if macro_bear and kama_bearish and rsi_overbought:
                desired_signal = -REDUCED_SIZE
            # Fallback: Extreme RSI with SMA200 filter
            if rsi_extreme_oversold and above_sma200 and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            if rsi_extreme_overbought and below_sma200 and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === FUNDING RATE OVERLAY (soft filter) ===
        if funding_zscore is not None and i < len(funding_zscore):
            if funding_bullish and desired_signal < 0:
                desired_signal = 0  # Don't short when funding extreme negative
            if funding_bearish and desired_signal > 0:
                desired_signal = 0  # Don't long when funding extreme positive
            # Boost signal if funding agrees
            if funding_bullish and desired_signal > 0:
                desired_signal = BASE_SIZE
            if funding_bearish and desired_signal < 0:
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro/KAMA trend intact
                if macro_bull and kama_bullish and rsi_1d[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro/KAMA trend intact
                if macro_bear and kama_bearish and rsi_1d[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro + KAMA reverses
            if macro_bear and kama_bearish:
                desired_signal = 0.0
            # Exit if RSI extremely overbought in ranging regime
            if ranging_regime and rsi_1d[i] > 75:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro + KAMA reverses
            if macro_bull and kama_bullish:
                desired_signal = 0.0
            # Exit if RSI extremely oversold in ranging regime
            if ranging_regime and rsi_1d[i] < 25:
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
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
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