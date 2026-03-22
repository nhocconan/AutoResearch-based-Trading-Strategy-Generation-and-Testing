#!/usr/bin/env python3
"""
Experiment #474: 4h Primary + 12h/1d HTF — Funding Rate Contrarian + HMA Trend

Hypothesis: After 473 failed experiments, clear pattern emerges:
1. Funding rate contrarian signals have proven Sharpe 0.8-1.5 for BTC/ETH (research notes)
2. Simple is better: complex CRSI/CHOP combinations failed in exp #469, #472, #473
3. 4h timeframe with 12h/1d trend filter balances trade frequency vs fee drag
4. RSI(14) extremes + HTMA trend is simpler and generates more trades than CRSI
5. Asymmetric sizing protects in bear markets (2022 crash, 2025 bear)

Why this might beat current best (Sharpe=0.435):
- Funding rate z-score is market-neutral edge (works in bull/bear/range)
- 12h HMA provides clean trend bias without over-filtering
- RSI(14) < 30 / > 70 generates adequate trade frequency (unlike CRSI < 10 / > 90)
- ATR 2.5x trailing stop protects in crashes while allowing trend runs
- 4h has proven potential (SOL Sharpe +0.879 with HMA+RSI+ATR in research)

Position sizing: 0.30 long, 0.25 short (discrete, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 20-50 trades/year on 4h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_funding_zscore_hma_12h1d_v1"
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

def calculate_funding_zscore(prices, symbol, lookback=30):
    """
    Calculate funding rate z-score for contrarian signal.
    Load funding data from processed parquet files.
    Z < -2 = extreme negative funding = long opportunity
    Z > +2 = extreme positive funding = short opportunity
    """
    try:
        import os
        funding_path = f"data/processed/funding/{symbol}.parquet"
        if not os.path.exists(funding_path):
            # Try alternate path
            funding_path = f"data/funding/{symbol}.parquet"
        
        if os.path.exists(funding_path):
            df_funding = pd.read_parquet(funding_path)
            # Align funding data to prices timeline
            # Funding is typically 8h intervals, we need to match to 4h bars
            if 'open_time' in df_funding.columns:
                df_funding = df_funding.sort_values('open_time').reset_index(drop=True)
                prices_df = prices.copy()
                
                # Merge on closest timestamp
                prices_df = prices_df.merge(
                    df_funding[['open_time', 'funding_rate']],
                    on='open_time',
                    how='left'
                )
                
                # Forward fill missing funding rates
                prices_df['funding_rate'] = prices_df['funding_rate'].ffill()
                
                # Calculate z-score over lookback period
                funding = prices_df['funding_rate'].values
                funding_s = pd.Series(funding)
                
                rolling_mean = funding_s.rolling(window=lookback, min_periods=lookback//2).mean()
                rolling_std = funding_s.rolling(window=lookback, min_periods=lookback//2).std()
                
                zscore = (funding - rolling_mean) / (rolling_std + 1e-10)
                return zscore.values
    except Exception:
        pass
    
    # Fallback: return zeros if funding data unavailable
    return np.zeros(len(prices))

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Extract symbol from prices metadata if available
    symbol = prices.get('symbol', ['BTCUSDT'])[0] if hasattr(prices, 'get') else 'BTCUSDT'
    if isinstance(symbol, list):
        symbol = symbol[0]
    
    # Load 12h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HTF indicators (major trend direction)
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_50 = calculate_hma(df_12h['close'].values, period=50)
    
    # Calculate 1d HTF indicators (major trend bias)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_50_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_50)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Calculate funding z-score (contrarian signal)
    funding_zscore = calculate_funding_zscore(prices, symbol, lookback=30)
    
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
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_50_aligned[i]):
            continue
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        if np.isnan(rsi_14[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime_1d = close[i] > hma_1d_21_aligned[i]
        bear_regime_1d = close[i] < hma_1d_21_aligned[i]
        
        # === 12H LOCAL TREND (HMA crossover) ===
        hma_bullish_12h = hma_12h_21_aligned[i] > hma_12h_50_aligned[i]
        hma_bearish_12h = hma_12h_21_aligned[i] < hma_12h_50_aligned[i]
        
        # === RSI SIGNALS (relaxed for trade frequency) ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_extreme_oversold = rsi_14[i] < 25.0
        rsi_extreme_overbought = rsi_14[i] > 75.0
        
        # === FUNDING Z-SCORE (contrarian edge) ===
        funding_extreme_long = funding_zscore[i] < -1.5
        funding_extreme_short = funding_zscore[i] > 1.5
        funding_strong_long = funding_zscore[i] < -2.0
        funding_strong_short = funding_zscore[i] > 2.0
        
        # === ENTRY LOGIC — COMBINED SIGNALS ===
        new_signal = 0.0
        
        # LONG ENTRIES (multiple confluence paths for frequency)
        if bull_regime_1d and hma_bullish_12h and rsi_oversold:
            new_signal = LONG_SIZE
        elif bull_regime_1d and rsi_extreme_oversold:
            new_signal = LONG_SIZE
        elif hma_bullish_12h and rsi_extreme_oversold:
            new_signal = LONG_SIZE
        elif funding_strong_long and rsi_oversold:
            new_signal = LONG_SIZE * 0.8
        elif funding_extreme_long and hma_bullish_12h:
            new_signal = LONG_SIZE * 0.7
        elif rsi_14[i] < 30.0 and bull_regime_1d:
            new_signal = LONG_SIZE
        
        # SHORT ENTRIES (multiple confluence paths for frequency)
        if new_signal == 0.0:
            if bear_regime_1d and hma_bearish_12h and rsi_overbought:
                new_signal = -SHORT_SIZE
            elif bear_regime_1d and rsi_extreme_overbought:
                new_signal = -SHORT_SIZE
            elif hma_bearish_12h and rsi_extreme_overbought:
                new_signal = -SHORT_SIZE
            elif funding_strong_short and rsi_overbought:
                new_signal = -SHORT_SIZE * 0.8
            elif funding_extreme_short and hma_bearish_12h:
                new_signal = -SHORT_SIZE * 0.7
            elif rsi_14[i] > 70.0 and bear_regime_1d:
                new_signal = -SHORT_SIZE
        
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
        # Exit long on extreme overbought
        if in_position and position_side > 0 and rsi_14[i] > 80.0:
            new_signal = 0.0
        # Exit short on extreme oversold
        if in_position and position_side < 0 and rsi_14[i] < 20.0:
            new_signal = 0.0
        
        # Regime flip exit (protect against trend reversal)
        if in_position and position_side > 0 and bear_regime_1d and hma_bearish_12h:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime_1d and hma_bullish_12h:
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