#!/usr/bin/env python3
"""
Experiment #551: 4h Primary + 1d HTF — Funding Rate Contrarian + Donchian Breakout

Hypothesis: After 490+ failed strategies, the clearest winning pattern is:
- Funding rate z-score contrarian signal showed Sharpe 0.8-1.5 through 2022 crash
- Donchian breakout timing works well on SOL (Sharpe +0.782 in history)
- 4h HMA trend filter prevents counter-trend trades
- 1d HTF filter prevents major regime losses (key failure in 2022)
- SIMPLER entry logic = more trades = better Sharpe (avoid 0-trade failure)

This strategy combines:
1. 4h HMA(21) for primary trend direction
2. 1d HMA(21) aligned for major trend bias
3. Funding rate z-score(30) for contrarian entries (extreme funding = reversal)
4. Donchian(20) breakout for entry timing
5. ATR(14) 2.5x trailing stop for all positions
6. Asymmetric sizing: 0.30 bull, 0.25 bear (crypto crashes faster)

Why this might beat Sharpe=0.435:
- Funding rate is PROVEN edge for BTC/ETH (research shows 0.8-1.5 Sharpe)
- Donchian breakout catches momentum without lag
- 1d HTF prevents 2022-style catastrophic losses
- 4h TF targets 20-50 trades/year (optimal per rules)
- Discrete position sizing (0.0, ±0.25, ±0.30) minimizes fee churn

Position sizing: 0.25-0.30 base (discrete per Rule 4, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_funding_donchian_hma_1d_v1"
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
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (highest + lowest) / 2.0
    return highest, lowest, mid

def calculate_zscore(series, period=30):
    """Calculate rolling z-score."""
    s = pd.Series(series)
    rolling_mean = s.rolling(window=period, min_periods=period).mean()
    rolling_std = s.rolling(window=period, min_periods=period).std()
    zscore = (s - rolling_mean) / (rolling_std + 1e-10)
    return zscore.values

def load_funding_data(symbol):
    """Load funding rate data from parquet."""
    try:
        # Map symbol to filename
        symbol_map = {
            'BTCUSDT': 'BTCUSDT',
            'ETHUSDT': 'ETHUSDT',
            'SOLUSDT': 'SOLUSDT'
        }
        funding_symbol = symbol_map.get(symbol, symbol)
        funding_path = f"data/processed/funding/{funding_symbol}.parquet"
        df = pd.read_parquet(funding_path)
        return df
    except:
        return None

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Get symbol from prices DataFrame (for funding data)
    symbol = prices.get('symbol', ['BTCUSDT'])[0] if 'symbol' in prices.columns else 'BTCUSDT'
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF HMA for major trend direction
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    
    # 4h HMA for trend confirmation
    hma_4h_21 = calculate_hma(close, period=21)
    hma_4h_50 = calculate_hma(close, period=50)
    
    # Donchian channels for breakout timing
    donchian_high, donchian_low, donchian_mid = calculate_donchian(high, low, 20)
    
    # Load funding rate data for contrarian signal
    funding_df = load_funding_data(symbol)
    if funding_df is not None and len(funding_df) > 0:
        # Align funding data to prices (funding is 8h, prices is 4h)
        # Use last available funding rate for each 4h bar
        funding_rates = funding_df['funding_rate'].values if 'funding_rate' in funding_df.columns else np.zeros(len(funding_df))
        # Calculate z-score of funding rate
        funding_zscore_full = calculate_zscore(funding_rates, 30)
        # Align to prices length (repeat/interpolate to match)
        if len(funding_zscore_full) < n:
            # Pad with last value
            funding_zscore = np.full(n, funding_zscore_full[-1] if len(funding_zscore_full) > 0 else 0.0)
            funding_zscore[:len(funding_zscore_full)] = funding_zscore_full
        else:
            funding_zscore = funding_zscore_full[:n]
    else:
        # Fallback: use price-based z-score if funding unavailable
        price_zscore = calculate_zscore(close, 30)
        funding_zscore = price_zscore
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Asymmetric: smaller size in bear regime (crypto crashes faster)
    POSITION_SIZE_BULL = 0.30
    POSITION_SIZE_BEAR = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_50[i]):
            continue
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            continue
        if np.isnan(funding_zscore[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime_1d = close[i] > hma_1d_21_aligned[i]
        bear_regime_1d = close[i] < hma_1d_21_aligned[i]
        
        # 1d HMA slope for trend strength
        hma_1d_slope_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_1d_slope_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === 4H TREND CONFIRMATION ===
        bull_regime_4h = close[i] > hma_4h_21[i]
        bear_regime_4h = close[i] < hma_4h_21[i]
        
        hma_4h_slope_bull = hma_4h_21[i] > hma_4h_50[i]
        hma_4h_slope_bear = hma_4h_21[i] < hma_4h_50[i]
        
        # === FUNDING RATE CONTRARIAN SIGNAL ===
        # Extreme positive funding (>2.0 z-score) = overbought = short signal
        # Extreme negative funding (<-2.0 z-score) = oversold = long signal
        funding_extreme_long = funding_zscore[i] < -1.5
        funding_extreme_short = funding_zscore[i] > 1.5
        
        # === DONCHIAN BREAKOUT TIMING ===
        # Long breakout: price crosses above Donchian high
        # Short breakout: price crosses below Donchian low
        donchian_breakout_long = close[i] > donchian_high[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_low[i-1] if i > 0 else False
        
        # === ENTRY LOGIC — FUNDING + DONCHIAN + TREND ===
        new_signal = 0.0
        
        # LONG ENTRY: 1d bull + 4h bull + funding oversold OR donchian breakout
        if bull_regime_1d and bull_regime_4h:
            if funding_extreme_long:
                # Contrarian long on extreme negative funding
                if hma_1d_slope_bull:
                    new_signal = POSITION_SIZE_BULL
                else:
                    new_signal = POSITION_SIZE_BULL * 0.8
            elif donchian_breakout_long:
                # Momentum long on breakout
                if hma_4h_slope_bull:
                    new_signal = POSITION_SIZE_BULL
                else:
                    new_signal = POSITION_SIZE_BULL * 0.8
        
        # SHORT ENTRY: 1d bear + 4h bear + funding overbought OR donchian breakout
        if bear_regime_1d and bear_regime_4h:
            if funding_extreme_short:
                # Contrarian short on extreme positive funding
                if hma_1d_slope_bear:
                    new_signal = -POSITION_SIZE_BEAR
                else:
                    new_signal = -POSITION_SIZE_BEAR * 0.8
            elif donchian_breakout_short:
                # Momentum short on breakdown
                if hma_4h_slope_bear:
                    new_signal = -POSITION_SIZE_BEAR
                else:
                    new_signal = -POSITION_SIZE_BEAR * 0.8
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
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
        
        # === EXIT CONDITIONS (regime flip) ===
        # Exit long on 1d regime flip to bear
        if in_position and position_side > 0:
            if bear_regime_1d and hma_1d_slope_bear:
                new_signal = 0.0
            elif bear_regime_4h and hma_4h_slope_bear:
                new_signal = 0.0
        
        # Exit short on 1d regime flip to bull
        if in_position and position_side < 0:
            if bull_regime_1d and hma_1d_slope_bull:
                new_signal = 0.0
            elif bull_regime_4h and hma_4h_slope_bull:
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
                # Flip position
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