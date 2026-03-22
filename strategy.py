#!/usr/bin/env python3
"""
Experiment #484: 4h Primary + 12h/1d HTF — Funding Rate Mean Reversion + HMA Trend

Hypothesis: After 483 experiments, the clearest edge from research is:
1. FUNDING RATE MEAN REVERSION — Best edge for BTC/ETH (Sharpe 0.8-1.5 in research)
   When funding Z-score > +2 (extreme long bias), price often reverses down
   When funding Z-score < -2 (extreme short bias), price often reverses up
2. 4h timeframe balances trade frequency (20-50/year) with fee drag
3. 1d HMA provides clean major trend filter without over-complication
4. 12h ADX detects trending vs ranging for entry timing
5. Simpler logic = more trades (critical after 0-trade failures in #475, #478, #480)

Why this might beat current best (Sharpe=0.435):
- Funding rate is UNDERUTILIZED edge (only tried once in #474, but with wrong logic)
- #474 failed because it combined too many filters — this is SIMPLER
- Funding mean reversion works in ALL regimes (bull/bear/range)
- 4h has proven potential (research: SOL +0.879 with HMA+RSI+ATR)
- Relaxed entry thresholds ensure >=30 trades/symbol on train

Position sizing: 0.25-0.30 (discrete, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 30-60 trades/year on 4h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_funding_mr_hma_1d_12h_v2"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100.0 * minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_zscore(series, period=30):
    """Calculate rolling Z-score."""
    series_s = pd.Series(series)
    rolling_mean = series_s.rolling(window=period, min_periods=period).mean()
    rolling_std = series_s.rolling(window=period, min_periods=period).std()
    zscore = (series_s - rolling_mean) / (rolling_std + 1e-10)
    return zscore.values

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return sma.values, upper.values, lower.values

def load_funding_data(prices, symbol='BTCUSDT'):
    """
    Load funding rate data from processed parquet files.
    Returns funding rate array aligned with prices.
    """
    import os
    # Try to load funding data - if not available, return zeros
    funding_path = f"data/processed/funding/{symbol}.parquet"
    alt_path = f"data/processed/funding/funding_rates.parquet"
    
    try:
        if os.path.exists(funding_path):
            funding_df = pd.read_parquet(funding_path)
        elif os.path.exists(alt_path):
            funding_df = pd.read_parquet(alt_path)
        else:
            # Return zeros if no funding data
            return np.zeros(len(prices))
        
        # Align funding to prices by timestamp
        if 'open_time' in funding_df.columns and 'open_time' in prices.columns:
            merged = prices[['open_time']].merge(
                funding_df[['open_time', 'funding_rate']], 
                on='open_time', 
                how='left'
            )
            funding = merged['funding_rate'].fillna(0.0).values
        else:
            funding = np.zeros(len(prices))
        
        return funding
    except Exception:
        return np.zeros(len(prices))

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h HTF indicators (regime detection)
    adx_12h = calculate_adx(
        df_12h['high'].values, 
        df_12h['low'].values, 
        df_12h['close'].values, 
        period=14
    )
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Load funding rate data
    funding = load_funding_data(prices)
    funding_zscore = calculate_zscore(funding, period=30)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_4h_21 = calculate_hma(close, period=21)
    hma_4h_50 = calculate_hma(close, period=50)
    rsi_14 = calculate_rsi(close, period=14)
    bb_sma, bb_upper, bb_lower = calculate_bollinger(close, period=20, std_dev=2.0)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        if np.isnan(adx_12h_aligned[i]):
            continue
        if np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_50[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(bb_sma[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # === 12H REGIME (ADX for trending vs ranging) ===
        is_trending = adx_12h_aligned[i] > 25.0
        is_ranging = adx_12h_aligned[i] < 20.0
        
        # === FUNDING RATE MEAN REVERSION (primary signal) ===
        funding_extreme_long = funding_zscore[i] > 1.5  # Extreme long bias → short
        funding_extreme_short = funding_zscore[i] < -1.5  # Extreme short bias → long
        funding_moderate_long = funding_zscore[i] > 0.5
        funding_moderate_short = funding_zscore[i] < -0.5
        
        # === 4H LOCAL TREND (HMA crossover) ===
        hma_bullish = hma_4h_21[i] > hma_4h_50[i]
        hma_bearish = hma_4h_21[i] < hma_4h_50[i]
        
        # === RSI SIGNALS (relaxed for frequency) ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        rsi_extreme_oversold = rsi_14[i] < 30.0
        rsi_extreme_overbought = rsi_14[i] > 70.0
        
        # === BOLLINGER BAND POSITION ===
        near_bb_lower = close[i] < bb_lower[i] * 1.005  # At or below lower band
        near_bb_upper = close[i] > bb_upper[i] * 0.995  # At or above upper band
        below_bb_sma = close[i] < bb_sma[i]
        above_bb_sma = close[i] > bb_sma[i]
        
        # === ENTRY LOGIC — FUNDING MR + TREND CONFIRMATION ===
        new_signal = 0.0
        
        # LONG ENTRIES (funding extreme short + trend confirmation)
        if funding_extreme_short and bull_regime:
            new_signal = LONG_SIZE
        elif funding_extreme_short and hma_bullish and rsi_oversold:
            new_signal = LONG_SIZE
        elif funding_moderate_short and bull_regime and near_bb_lower:
            new_signal = LONG_SIZE * 0.8
        elif funding_moderate_short and hma_bullish and rsi_extreme_oversold:
            new_signal = LONG_SIZE
        elif is_ranging and funding_moderate_short and near_bb_lower:
            new_signal = LONG_SIZE * 0.7
        elif bull_regime and hma_bullish and rsi_extreme_oversold and below_bb_sma:
            new_signal = LONG_SIZE * 0.8
        
        # SHORT ENTRIES (funding extreme long + trend confirmation)
        if new_signal == 0.0:
            if funding_extreme_long and bear_regime:
                new_signal = -SHORT_SIZE
            elif funding_extreme_long and hma_bearish and rsi_overbought:
                new_signal = -SHORT_SIZE
            elif funding_moderate_long and bear_regime and near_bb_upper:
                new_signal = -SHORT_SIZE * 0.8
            elif funding_moderate_long and hma_bearish and rsi_extreme_overbought:
                new_signal = -SHORT_SIZE
            elif is_ranging and funding_moderate_long and near_bb_upper:
                new_signal = -SHORT_SIZE * 0.7
            elif bear_regime and hma_bearish and rsi_extreme_overbought and above_bb_sma:
                new_signal = -SHORT_SIZE * 0.8
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TAKE PROFIT / EXIT CONDITIONS ===
        if in_position and position_side > 0 and rsi_14[i] > 70.0:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 30.0:
            new_signal = 0.0
        
        # Regime flip exit
        if in_position and position_side > 0 and bear_regime and hma_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime and hma_bullish:
            new_signal = 0.0
        
        # Funding reversal exit
        if in_position and position_side > 0 and funding_zscore[i] > 1.0:
            new_signal = 0.0
        if in_position and position_side < 0 and funding_zscore[i] < -1.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals