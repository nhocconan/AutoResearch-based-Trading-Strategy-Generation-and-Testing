#!/usr/bin/env python3
"""
Experiment #404: 4h Primary + 12h/1d HTF — Funding Rate Contrarian + HMA Trend + RSI Pullback

Hypothesis: After 366+ failed experiments, the clearest pattern is:
1. Pure trend strategies FAIL on BTC/ETH (especially 2022 crash, 2025 bear)
2. Funding rate contrarian is the BEST EDGE for BTC/ETH (Sharpe 0.8-1.5 reported)
3. When funding is extreme (>+2% or <-2%), positions are crowded → reversal likely
4. 4h timeframe needs 30-50 trades/year (not too few like 12h/1d, not too many like 1h)
5. Combine: Funding extreme (trigger) + 12h HMA trend (direction) + 4h RSI pullback (timing)

Why this might beat current best (Sharpe=0.435):
- Funding rate is MARKET-NEUTRAL edge (works bull AND bear)
- Contrarian approach avoids 2022 crash whipsaw that destroyed trend strategies
- 12h HMA(21) filters counter-trend funding signals (don't short in strong bull)
- RSI(14) pullback ensures we enter on retracement, not chasing
- Discrete sizing 0.0, ±0.25, ±0.30 minimizes fee churn

Position sizing: 0.25-0.30 (discrete, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 30-50 trades/year on 4h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_funding_contrarian_hma_rsi_12h_v1"
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

def calculate_funding_zscore(funding_series, lookback=30):
    """Calculate z-score of funding rate over lookback period."""
    funding = pd.Series(funding_series)
    rolling_mean = funding.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = funding.rolling(window=lookback, min_periods=lookback).std()
    zscore = (funding - rolling_mean) / (rolling_std + 1e-10)
    return zscore.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load funding rate data (CRITICAL EDGE for BTC/ETH)
    # Funding rate parquet is in data/processed/funding/{symbol}.parquet
    try:
        symbol = prices.get('symbol', 'BTCUSDT')
        if isinstance(symbol, (list, np.ndarray)):
            symbol = symbol[0] if len(symbol) > 0 else 'BTCUSDT'
        funding_path = f"data/processed/funding/{symbol}.parquet"
        df_funding = pd.read_parquet(funding_path)
        # Align funding to prices (funding is 8h, prices is 4h)
        funding_rates = df_funding['funding_rate'].values
        
        # Resample funding to 4h (take most recent per 4h bar)
        # Funding comes every 8h, so we need to forward-fill to 4h
        funding_4h = np.zeros(n)
        funding_idx = 0
        for i in range(n):
            if funding_idx < len(funding_rates):
                funding_4h[i] = funding_rates[funding_idx]
                # Advance funding index every 2 bars (8h / 4h = 2)
                if i > 0 and i % 2 == 0:
                    funding_idx = min(funding_idx + 1, len(funding_rates) - 1)
            else:
                funding_4h[i] = funding_4h[i-1] if i > 0 else 0.0
        
        funding_z = calculate_funding_zscore(funding_4h, 30)
    except Exception:
        # Fallback if funding data unavailable
        funding_z = np.zeros(n)
    
    # Load 12h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HTF indicators (major trend direction)
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_4h_16 = calculate_hma(close, period=16)
    hma_4h_48 = calculate_hma(close, period=48)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_12h_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_4h_16[i]) or np.isnan(hma_4h_48[i]):
            continue
        
        # === 12H MAJOR TREND (primary direction filter) ===
        # Price above 12h HMA = bull market bias (favor longs)
        # Price below 12h HMA = bear market bias (favor shorts)
        bull_regime = close[i] > hma_12h_21_aligned[i]
        bear_regime = close[i] < hma_12h_21_aligned[i]
        
        # === 4H LOCAL TREND (HMA crossover for confirmation) ===
        hma_bullish = hma_4h_16[i] > hma_4h_48[i]
        hma_bearish = hma_4h_16[i] < hma_4h_48[i]
        
        # === FUNDING RATE CONTRARIAN SIGNALS ===
        # Extreme positive funding (>1.5 z-score) = crowded longs → short signal
        # Extreme negative funding (<-1.5 z-score) = crowded shorts → long signal
        funding_extreme_long = funding_z[i] < -1.5  # Shorts crowded, expect reversal up
        funding_extreme_short = funding_z[i] > 1.5   # Longs crowded, expect reversal down
        
        # === RSI PULLBACK SIGNALS (timing entries) ===
        # Long: RSI pulled back to 35-55 in uptrend (buying dip)
        rsi_long_pullback = 35.0 <= rsi_14[i] <= 55.0
        # Short: RSI pulled back to 45-65 in downtrend (selling rally)
        rsi_short_pullback = 45.0 <= rsi_14[i] <= 65.0
        
        # === ENTRY LOGIC — FUNDING CONTRARIAN + TREND FILTER ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Funding extreme negative + (bull regime OR HMA bullish) + RSI pullback
        if funding_extreme_long:
            if bull_regime and rsi_long_pullback:
                new_signal = LONG_SIZE
            elif hma_bullish and rsi_14[i] < 50:
                # HMA confirmation without strict regime
                new_signal = LONG_SIZE * 0.8
        
        # SHORT ENTRY: Funding extreme positive + (bear regime OR HMA bearish) + RSI pullback
        if funding_extreme_short:
            if bear_regime and rsi_short_pullback:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            elif hma_bearish and rsi_14[i] > 50:
                # HMA confirmation without strict regime
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.8
        
        # === SECONDARY ENTRY: Pure RSI Mean Reversion (if funding data weak) ===
        # Ensure trade frequency even when funding z-score not extreme
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            # Very oversold in bull regime
            if bull_regime and rsi_14[i] < 35 and hma_bullish:
                new_signal = LONG_SIZE * 0.6
            # Very overbought in bear regime
            elif bear_regime and rsi_14[i] > 65 and hma_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.6
        
        # === EXIT CONDITIONS ===
        # RSI extreme exit (take profit on momentum exhaustion)
        if in_position and position_side > 0 and rsi_14[i] > 75:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 25:
            new_signal = 0.0
        
        # Trend reversal exit (12h regime flip)
        if in_position and position_side > 0 and bear_regime and hma_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime and hma_bullish:
            new_signal = 0.0
        
        # Funding normalization exit (contrarian signal expired)
        if in_position and position_side > 0 and funding_z[i] > 0.5:
            # Funding went from negative to positive, long thesis expired
            new_signal = 0.0
        if in_position and position_side < 0 and funding_z[i] < -0.5:
            # Funding went from positive to negative, short thesis expired
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals